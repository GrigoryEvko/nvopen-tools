// Function: sub_16913D0
// Address: 0x16913d0
//
unsigned __int64 __fastcall sub_16913D0(char **a1, const void *a2, __int64 a3, char a4)
{
  __int64 v4; // rdx
  __int64 v5; // rbx
  size_t v8; // r12
  char *v9; // rax
  char *v10; // r13
  char *v11; // rcx
  unsigned __int64 result; // rax
  char *v13; // rax
  int v14; // r12d
  int v15; // edx
  int v16; // esi
  int v17; // edx
  const char **v18; // rbx
  const char *v19; // r14
  size_t v20; // r12
  unsigned int v21; // r8d
  unsigned __int64 *v22; // rcx
  __int64 v23; // rax
  unsigned int v24; // r8d
  _QWORD *v25; // rcx
  __int64 v26; // r13
  unsigned __int64 *v27; // r12
  __int64 v28; // rdx
  __int64 v29; // rdx
  char **v30; // r14
  __int64 v31; // r13
  char **v32; // rax
  unsigned __int64 *v33; // r9
  char *v34; // r14
  char *v35; // r12
  __int64 v36; // r13
  char *v37; // rsi
  char v38; // bl
  char *v39; // rdi
  __int64 v40; // rcx
  char *v41; // rdx
  unsigned __int64 v42; // rdx
  __int64 v43; // rdx
  unsigned __int64 v44; // r10
  char *v45; // rdx
  int v46; // [rsp+1Ch] [rbp-54h]
  unsigned int v47; // [rsp+20h] [rbp-50h]
  unsigned __int64 *v48; // [rsp+20h] [rbp-50h]
  unsigned __int64 *v49; // [rsp+28h] [rbp-48h]
  _QWORD *v50; // [rsp+28h] [rbp-48h]
  unsigned __int64 *v51; // [rsp+28h] [rbp-48h]
  unsigned int v52; // [rsp+30h] [rbp-40h]
  unsigned int v53; // [rsp+30h] [rbp-40h]
  char **v54; // [rsp+30h] [rbp-40h]
  void *src; // [rsp+38h] [rbp-38h]
  void *srca; // [rsp+38h] [rbp-38h]

  v4 = a3 << 6;
  v5 = v4 >> 6;
  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  if ( (unsigned __int64)v4 > 0x7FFFFFFFFFFFFFC0LL )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v8 = v4;
  if ( v4 )
  {
    v9 = (char *)sub_22077B0(v4);
    v10 = &v9[v8];
    *a1 = v9;
    a1[2] = &v9[v8];
    v11 = (char *)memcpy(v9, a2, v8);
  }
  else
  {
    v10 = 0;
    v11 = 0;
  }
  a1[1] = v10;
  a1[7] = (char *)0x1000000000LL;
  result = (unsigned __int64)(a1 + 11);
  *((_BYTE *)a1 + 24) = a4;
  *((_DWORD *)a1 + 7) = 0;
  a1[4] = 0;
  a1[5] = 0;
  a1[6] = 0;
  a1[9] = (char *)(a1 + 11);
  a1[10] = 0;
  *((_BYTE *)a1 + 88) = 0;
  if ( !(_DWORD)v5 )
    return result;
  v13 = v11 + 32;
  v14 = 0;
  while ( 1 )
  {
    while ( 1 )
    {
      v15 = (unsigned __int8)v13[4];
      v16 = v14++;
      if ( v15 != 1 )
        break;
      v17 = *(_DWORD *)v13;
      v13 += 64;
      *((_DWORD *)a1 + 7) = v17;
      if ( v14 == (_DWORD)v5 )
      {
LABEL_11:
        LODWORD(result) = v14 + 1;
        v14 = 1;
        goto LABEL_12;
      }
    }
    if ( v15 == 2 )
    {
      *((_DWORD *)a1 + 8) = *(_DWORD *)v13;
      goto LABEL_8;
    }
    if ( v13[4] )
      break;
LABEL_8:
    v13 += 64;
    if ( v14 == (_DWORD)v5 )
      goto LABEL_11;
  }
  *((_DWORD *)a1 + 9) = v16;
  result = (unsigned int)(v5 + 1);
  if ( (_DWORD)v5 == v16 )
    return result;
LABEL_12:
  v47 = v14 - 1;
  v46 = result - 2;
  while ( 2 )
  {
    result = (unsigned __int64)v47 << 6;
    v18 = *(const char ***)&v11[result];
    if ( v18 )
    {
      v19 = *v18;
      if ( *v18 )
      {
        src = a1 + 5;
        while ( 1 )
        {
          v20 = strlen(v19);
          v21 = sub_16D19C0(src, v19, v20);
          v22 = (unsigned __int64 *)&a1[5][8 * v21];
          result = *v22;
          if ( !*v22 )
            goto LABEL_22;
          if ( result == -8 )
            break;
LABEL_18:
          v19 = v18[1];
          ++v18;
          if ( !v19 )
            goto LABEL_13;
        }
        --*((_DWORD *)a1 + 14);
LABEL_22:
        v49 = v22;
        v52 = v21;
        v23 = malloc(v20 + 17);
        v24 = v52;
        v25 = v49;
        v26 = v23;
        if ( !v23 )
        {
          sub_16BD1C0("Allocation failed");
          v25 = v49;
          v24 = v52;
        }
        if ( v20 )
        {
          v50 = v25;
          v53 = v24;
          memcpy((void *)(v26 + 16), v19, v20);
          v25 = v50;
          v24 = v53;
        }
        *(_BYTE *)(v26 + v20 + 16) = 0;
        *(_QWORD *)v26 = v20;
        *(_BYTE *)(v26 + 8) = 0;
        *v25 = v26;
        ++*((_DWORD *)a1 + 13);
        result = sub_16D1CD0(src, v24);
        goto LABEL_18;
      }
    }
LABEL_13:
    if ( v46 != v47 )
    {
      ++v47;
      v11 = *a1;
      continue;
    }
    break;
  }
  v27 = (unsigned __int64 *)a1[5];
  v28 = *((unsigned int *)a1 + 12);
  v48 = &v27[v28];
  if ( (_DWORD)v28 )
  {
    result = *v27;
    if ( !*v27 || result == -8 )
    {
      result = (unsigned __int64)(v27 + 1);
      do
      {
        do
        {
          v29 = *(_QWORD *)result;
          v27 = (unsigned __int64 *)result;
          result += 8LL;
        }
        while ( v29 == -8 );
      }
      while ( !v29 );
    }
  }
  if ( v48 != v27 )
  {
    v30 = a1 + 11;
    while ( 1 )
    {
      v31 = *v27 + 16;
      if ( v31 + *(_QWORD *)*v27 != v31 )
        break;
LABEL_50:
      v42 = v27[1];
      result = (unsigned __int64)(v27 + 1);
      if ( v42 == -8 || !v42 )
      {
        do
        {
          do
          {
            v43 = *(_QWORD *)(result + 8);
            result += 8LL;
          }
          while ( !v43 );
        }
        while ( v43 == -8 );
      }
      if ( (unsigned __int64 *)result == v48 )
        return result;
      v27 = (unsigned __int64 *)result;
    }
    v32 = v30;
    v33 = v27;
    v34 = (char *)(v31 + *(_QWORD *)*v27);
    v35 = (char *)(*v27 + 16);
    while ( 2 )
    {
      v36 = (__int64)a1[10];
      v37 = a1[9];
      v38 = *v35;
      v39 = &v37[v36];
      v40 = v36;
      if ( v36 >> 2 > 0 )
      {
        v41 = a1[9];
        while ( v38 != *v41 )
        {
          if ( v38 == v41[1] )
          {
            ++v41;
            break;
          }
          if ( v38 == v41[2] )
          {
            v41 += 2;
            break;
          }
          if ( v38 == v41[3] )
          {
            v41 += 3;
            break;
          }
          v41 += 4;
          if ( v41 == &v37[4 * (v36 >> 2)] )
          {
            v40 = v39 - v41;
            goto LABEL_57;
          }
        }
LABEL_47:
        if ( v39 != v41 )
        {
LABEL_48:
          if ( ++v35 == v34 )
          {
            v30 = v32;
            v27 = v33;
            goto LABEL_50;
          }
          continue;
        }
LABEL_60:
        v44 = v36 + 1;
        if ( v32 != (char **)v37 )
        {
LABEL_61:
          if ( v44 <= (unsigned __int64)a1[11] )
          {
LABEL_62:
            *v39 = v38;
            v45 = a1[9];
            a1[10] = (char *)v44;
            v45[v36 + 1] = 0;
            goto LABEL_48;
          }
LABEL_73:
          v51 = v33;
          v54 = v32;
          srca = (void *)v44;
          sub_2240BB0(a1 + 9, a1[10], 0, 0, 1);
          v33 = v51;
          v32 = v54;
          v44 = (unsigned __int64)srca;
          v39 = &a1[9][v36];
          goto LABEL_62;
        }
LABEL_72:
        if ( v44 <= 0xF )
          goto LABEL_62;
        goto LABEL_73;
      }
      break;
    }
    v41 = a1[9];
LABEL_57:
    if ( v40 != 2 )
    {
      if ( v40 != 3 )
      {
        if ( v40 != 1 )
          goto LABEL_60;
LABEL_70:
        if ( v38 == *v41 )
          goto LABEL_47;
        v44 = v36 + 1;
        if ( v32 != (char **)v37 )
          goto LABEL_61;
        goto LABEL_72;
      }
      if ( v38 == *v41 )
        goto LABEL_47;
      ++v41;
    }
    if ( v38 == *v41 )
      goto LABEL_47;
    ++v41;
    goto LABEL_70;
  }
  return result;
}
