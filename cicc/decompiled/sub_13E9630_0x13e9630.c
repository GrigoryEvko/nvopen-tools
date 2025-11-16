// Function: sub_13E9630
// Address: 0x13e9630
//
int *__fastcall sub_13E9630(int *a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v6; // al
  unsigned int v8; // eax
  __int64 v9; // rax
  __int64 v11; // rsi
  int v12; // r9d
  unsigned int v13; // edx
  __int64 *v14; // r13
  __int64 v15; // rdi
  _QWORD *v16; // rdx
  _QWORD *v17; // rax
  _QWORD *v18; // r15
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdi
  unsigned int v22; // esi
  __int64 *v23; // rdx
  __int64 v24; // r9
  __int64 v25; // r8
  char v26; // di
  __int64 v27; // rdx
  int v28; // r9d
  unsigned int v29; // eax
  unsigned int *v30; // rsi
  __int64 v31; // r10
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rsi
  int v36; // eax
  unsigned int v37; // eax
  __int64 v38; // rdi
  __int64 v39; // rdi
  __int64 v40; // rsi
  __int64 v41; // rdi
  bool v42; // cc
  unsigned int v43; // eax
  __int64 v44; // rdi
  int v45; // edx
  int v46; // r10d
  int v47; // esi
  _QWORD *v48; // rdx
  int v49; // r11d
  __int64 v50; // [rsp+8h] [rbp-68h]
  __int64 v51; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v52; // [rsp+18h] [rbp-58h]
  __int64 v53; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v54; // [rsp+28h] [rbp-48h]
  __int64 v55; // [rsp+30h] [rbp-40h]
  unsigned int v56; // [rsp+38h] [rbp-38h]

  if ( *(_BYTE *)(a3 + 16) <= 0x10u )
  {
    *a1 = 0;
    v6 = *(_BYTE *)(a3 + 16);
    if ( v6 == 9 )
      return a1;
    if ( v6 != 13 )
    {
      *a1 = 1;
      *((_QWORD *)a1 + 1) = a3;
      return a1;
    }
    v52 = *(_DWORD *)(a3 + 32);
    if ( v52 > 0x40 )
      sub_16A4FD0(&v51, a3 + 24);
    else
      v51 = *(_QWORD *)(a3 + 24);
    sub_1589870(&v53, &v51);
    if ( *a1 == 3 )
    {
      if ( !(unsigned __int8)sub_158A120(&v53) )
      {
        if ( (unsigned int)a1[4] > 0x40 )
        {
          v41 = *((_QWORD *)a1 + 1);
          if ( v41 )
            j_j___libc_free_0_0(v41);
        }
        v42 = (unsigned int)a1[8] <= 0x40;
        *((_QWORD *)a1 + 1) = v53;
        v43 = v54;
        v54 = 0;
        a1[4] = v43;
        if ( v42 || (v44 = *((_QWORD *)a1 + 3)) == 0 )
        {
          *((_QWORD *)a1 + 3) = v55;
          a1[8] = v56;
          goto LABEL_10;
        }
        j_j___libc_free_0_0(v44);
        v37 = v54;
        *((_QWORD *)a1 + 3) = v55;
        a1[8] = v56;
LABEL_55:
        if ( v37 > 0x40 && v53 )
          j_j___libc_free_0_0(v53);
        goto LABEL_10;
      }
    }
    else if ( !(unsigned __int8)sub_158A120(&v53) )
    {
      v8 = v54;
      *a1 = 3;
      a1[4] = v8;
      *((_QWORD *)a1 + 1) = v53;
      a1[8] = v56;
      *((_QWORD *)a1 + 3) = v55;
LABEL_10:
      if ( v52 > 0x40 && v51 )
        j_j___libc_free_0_0(v51);
      return a1;
    }
    v36 = *a1;
    if ( *a1 != 4 )
    {
      if ( (unsigned int)(v36 - 1) > 1 )
      {
        if ( v36 == 3 )
        {
          if ( (unsigned int)a1[8] > 0x40 )
          {
            v38 = *((_QWORD *)a1 + 3);
            if ( v38 )
              j_j___libc_free_0_0(v38);
          }
          if ( (unsigned int)a1[4] > 0x40 )
          {
            v39 = *((_QWORD *)a1 + 1);
            if ( v39 )
              j_j___libc_free_0_0(v39);
          }
        }
      }
      else
      {
        *((_QWORD *)a1 + 1) = 0;
      }
      *a1 = 4;
    }
    if ( v56 > 0x40 && v55 )
      j_j___libc_free_0_0(v55);
    v37 = v54;
    goto LABEL_55;
  }
  v9 = *(unsigned int *)(a2 + 88);
  if ( !(_DWORD)v9 )
    goto LABEL_23;
  v11 = *(_QWORD *)(a2 + 72);
  v12 = 1;
  v13 = (v9 - 1) & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
  v14 = (__int64 *)(v11 + 80LL * v13);
  v15 = *v14;
  if ( a4 == *v14 )
  {
LABEL_15:
    if ( v14 != (__int64 *)(v11 + 80 * v9) )
    {
      v16 = (_QWORD *)v14[3];
      v17 = (_QWORD *)v14[2];
      if ( v16 == v17 )
      {
        v18 = &v17[*((unsigned int *)v14 + 9)];
        if ( v17 == v18 )
        {
          v48 = (_QWORD *)v14[2];
        }
        else
        {
          do
          {
            if ( a3 == *v17 )
              break;
            ++v17;
          }
          while ( v18 != v17 );
          v48 = v18;
        }
      }
      else
      {
        v50 = a4;
        v18 = &v16[*((unsigned int *)v14 + 8)];
        v17 = (_QWORD *)sub_16CC9F0(v14 + 1, a3);
        a4 = v50;
        if ( a3 == *v17 )
        {
          v34 = v14[3];
          if ( v34 == v14[2] )
            v35 = *((unsigned int *)v14 + 9);
          else
            v35 = *((unsigned int *)v14 + 8);
          v48 = (_QWORD *)(v34 + 8 * v35);
        }
        else
        {
          v19 = v14[3];
          if ( v19 != v14[2] )
          {
            v17 = (_QWORD *)(v19 + 8LL * *((unsigned int *)v14 + 8));
            goto LABEL_20;
          }
          v17 = (_QWORD *)(v19 + 8LL * *((unsigned int *)v14 + 9));
          v48 = v17;
        }
      }
      while ( v48 != v17 && *v17 >= 0xFFFFFFFFFFFFFFFELL )
        ++v17;
LABEL_20:
      if ( v18 != v17 )
      {
        *a1 = 4;
        return a1;
      }
    }
  }
  else
  {
    while ( v15 != -8 )
    {
      v13 = (v9 - 1) & (v12 + v13);
      v14 = (__int64 *)(v11 + 80LL * v13);
      v15 = *v14;
      if ( a4 == *v14 )
        goto LABEL_15;
      ++v12;
    }
  }
LABEL_23:
  v20 = *(unsigned int *)(a2 + 56);
  if ( !(_DWORD)v20 )
    goto LABEL_58;
  v21 = *(_QWORD *)(a2 + 40);
  v22 = (v20 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v23 = (__int64 *)(v21 + 16LL * v22);
  v24 = *v23;
  if ( a3 != *v23 )
  {
    v45 = 1;
    while ( v24 != -8 )
    {
      v46 = v45 + 1;
      v22 = (v20 - 1) & (v45 + v22);
      v23 = (__int64 *)(v21 + 16LL * v22);
      v24 = *v23;
      if ( a3 == *v23 )
        goto LABEL_25;
      v45 = v46;
    }
    goto LABEL_58;
  }
LABEL_25:
  if ( v23 == (__int64 *)(v21 + 16 * v20) )
  {
LABEL_58:
    *a1 = 0;
    return a1;
  }
  v25 = v23[1];
  v26 = *(_BYTE *)(v25 + 48) & 1;
  if ( v26 )
  {
    v27 = v25 + 56;
    v28 = 3;
  }
  else
  {
    v33 = *(unsigned int *)(v25 + 64);
    v27 = *(_QWORD *)(v25 + 56);
    if ( !(_DWORD)v33 )
      goto LABEL_68;
    v28 = v33 - 1;
  }
  v29 = v28 & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
  v30 = (unsigned int *)(v27 + 48LL * v29);
  v31 = *(_QWORD *)v30;
  if ( a4 != *(_QWORD *)v30 )
  {
    v47 = 1;
    while ( v31 != -8 )
    {
      v49 = v47 + 1;
      v29 = v28 & (v47 + v29);
      v30 = (unsigned int *)(v27 + 48LL * v29);
      v31 = *(_QWORD *)v30;
      if ( a4 == *(_QWORD *)v30 )
        goto LABEL_29;
      v47 = v49;
    }
    if ( v26 )
    {
      v40 = 192;
      goto LABEL_69;
    }
    v33 = *(unsigned int *)(v25 + 64);
LABEL_68:
    v40 = 48 * v33;
LABEL_69:
    v30 = (unsigned int *)(v27 + v40);
  }
LABEL_29:
  v32 = 192;
  if ( !v26 )
    v32 = 48LL * *(unsigned int *)(v25 + 64);
  *a1 = 0;
  if ( v30 != (unsigned int *)(v27 + v32) )
    sub_13E8810(a1, v30 + 2);
  return a1;
}
