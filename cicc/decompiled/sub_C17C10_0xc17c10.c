// Function: sub_C17C10
// Address: 0xc17c10
//
__int64 __fastcall sub_C17C10(_QWORD *a1, __int64 *a2, __int64 *a3, __int64 a4)
{
  _QWORD *v4; // r13
  __int64 v6; // r10
  __int64 v7; // r9
  __int64 v9; // rsi
  __int64 v10; // rcx
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r12
  unsigned __int64 v15; // rax
  __int64 v16; // rcx
  __int64 v17; // rsi
  __int64 v18; // rdx
  _QWORD *v19; // rax
  __int64 v20; // r13
  _QWORD *v21; // r15
  int *v22; // rdx
  int v23; // r12d
  __int64 v24; // rax
  int *v25; // rsi
  __int64 v26; // rdx
  __int64 v27; // rdi
  unsigned int v28; // esi
  int *v29; // rax
  int v30; // r9d
  const void *v31; // r9
  signed __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rdi
  bool v35; // cf
  unsigned __int64 v36; // rax
  __int64 v37; // r12
  __int64 v38; // rax
  char *v39; // r10
  __int64 v40; // r12
  int v42; // eax
  char *v43; // rax
  __int64 v44; // rsi
  unsigned __int64 v45; // rdx
  unsigned __int64 v46; // rsi
  __int64 v47; // rcx
  int v48; // ecx
  size_t n; // [rsp+0h] [rbp-80h]
  void *src; // [rsp+8h] [rbp-78h]
  const void *v51; // [rsp+10h] [rbp-70h]
  char *v52; // [rsp+10h] [rbp-70h]
  __int64 v53; // [rsp+18h] [rbp-68h]
  __int64 v55; // [rsp+28h] [rbp-58h]
  unsigned __int64 v56; // [rsp+30h] [rbp-50h]
  unsigned __int64 v57; // [rsp+30h] [rbp-50h]
  __int64 v58; // [rsp+38h] [rbp-48h]
  int v59; // [rsp+48h] [rbp-38h] BYREF
  _DWORD v60[13]; // [rsp+4Ch] [rbp-34h] BYREF

  v4 = a1;
  v6 = a1[8];
  v7 = a1[7];
  v58 = (__int64)(a1 + 7);
  if ( a3 )
  {
    v9 = *a2;
    v10 = *a3;
    v11 = v9 + 4LL * *((unsigned int *)a2 + 2);
    v12 = *a3 + 4LL * *((unsigned int *)a3 + 2);
    if ( v12 == *a3 )
    {
      LODWORD(v14) = 0;
      v16 = 0;
      v15 = 0;
      v46 = (v6 - v7) >> 2;
    }
    else
    {
      v13 = v11;
      do
      {
        if ( v9 == v13 )
        {
          v14 = (v11 - v9) >> 2;
          v15 = (unsigned int)v14;
          v16 = 4LL * (unsigned int)v14;
          goto LABEL_44;
        }
        if ( *(_DWORD *)(v12 - 4) != *(_DWORD *)(v13 - 4) )
          break;
        v12 -= 4;
        v13 -= 4;
      }
      while ( v10 != v12 );
      v14 = (v11 - v13) >> 2;
      v15 = (unsigned int)v14;
      v16 = 4LL * (unsigned int)v14;
LABEL_44:
      v45 = (v6 - v7) >> 2;
      v46 = v45;
      if ( v15 > v45 )
      {
        v56 = v15;
        sub_C17A60(v58, v15 - v45);
        v15 = v56;
LABEL_46:
        if ( (_DWORD)v14 )
        {
          v57 = v15;
          v60[0] = *(_DWORD *)(v4[8] - 4LL) - ((__int64)(v4[1] - *v4) >> 2);
          sub_C15FF0((__int64)v4, v60);
          v15 = v57;
        }
        goto LABEL_11;
      }
    }
    if ( v46 > v15 )
    {
      v47 = v7 + v16;
      if ( v47 != v6 )
        v4[8] = v47;
    }
    goto LABEL_46;
  }
  if ( v7 != v6 )
    a1[8] = v7;
  v15 = 0;
LABEL_11:
  v17 = *((unsigned int *)a2 + 2);
  v18 = v17 - v15;
  v55 = *a2;
  if ( *a2 + 4 * (v17 - v15) != *a2 )
  {
    v19 = v4;
    v20 = *a2 + 4 * v18;
    v21 = v19;
    while ( 1 )
    {
      v23 = *(_DWORD *)(v20 - 4);
      v24 = (__int64)(v21[1] - *v21) >> 2;
      v59 = v23;
      v60[0] = v24;
      sub_C15FF0(v58, v60);
      v25 = &v59;
      if ( a4 )
        break;
LABEL_20:
      v22 = (int *)v21[1];
      if ( v22 == (int *)v21[2] )
      {
        v31 = (const void *)*v21;
        v32 = (signed __int64)v22 - *v21;
        v33 = v32 >> 2;
        if ( v32 >> 2 == 0x1FFFFFFFFFFFFFFFLL )
          sub_4262D8((__int64)"vector::_M_realloc_insert");
        v34 = 1;
        if ( v33 )
          v34 = v32 >> 2;
        v35 = __CFADD__(v34, v33);
        v36 = v34 + v33;
        if ( v35 )
        {
          v37 = 0x7FFFFFFFFFFFFFFCLL;
LABEL_29:
          n = v32;
          src = (void *)*v21;
          v38 = sub_22077B0(v37);
          v31 = src;
          v39 = (char *)v38;
          v32 = n;
          v53 = v37 + v38;
          goto LABEL_30;
        }
        if ( v36 )
        {
          if ( v36 > 0x1FFFFFFFFFFFFFFFLL )
            v36 = 0x1FFFFFFFFFFFFFFFLL;
          v37 = 4 * v36;
          goto LABEL_29;
        }
        v53 = 0;
        v39 = 0;
LABEL_30:
        if ( &v39[v32] )
          *(_DWORD *)&v39[v32] = *v25;
        v40 = (__int64)&v39[v32 + 4];
        if ( v32 > 0 )
        {
          v51 = v31;
          v43 = (char *)memmove(v39, v31, v32);
          v31 = v51;
          v39 = v43;
          v44 = v21[2] - (_QWORD)v51;
LABEL_41:
          v52 = v39;
          j_j___libc_free_0(v31, v44);
          v39 = v52;
          goto LABEL_34;
        }
        if ( v31 )
        {
          v44 = v21[2] - (_QWORD)v31;
          goto LABEL_41;
        }
LABEL_34:
        *v21 = v39;
        v20 -= 4;
        v21[1] = v40;
        v21[2] = v53;
        if ( v20 == v55 )
          goto LABEL_35;
      }
      else
      {
        if ( v22 )
        {
          *v22 = *v25;
          v22 = (int *)v21[1];
        }
        v20 -= 4;
        v21[1] = v22 + 1;
        if ( v20 == v55 )
        {
LABEL_35:
          v4 = v21;
          LODWORD(v17) = *((_DWORD *)a2 + 2);
          goto LABEL_36;
        }
      }
    }
    v26 = *(unsigned int *)(a4 + 24);
    v27 = *(_QWORD *)(a4 + 8);
    if ( (_DWORD)v26 )
    {
      v28 = (v26 - 1) & (37 * v23);
      v29 = (int *)(v27 + 8LL * v28);
      v30 = *v29;
      if ( v23 == *v29 )
      {
LABEL_19:
        v25 = v29 + 1;
        goto LABEL_20;
      }
      v42 = 1;
      while ( v30 != -1 )
      {
        v48 = v42 + 1;
        v28 = (v26 - 1) & (v42 + v28);
        v29 = (int *)(v27 + 8LL * v28);
        v30 = *v29;
        if ( v23 == *v29 )
          goto LABEL_19;
        v42 = v48;
      }
    }
    v29 = (int *)(v27 + 8 * v26);
    goto LABEL_19;
  }
LABEL_36:
  v60[0] = v17;
  sub_C15FF0((__int64)v4, v60);
  return (unsigned int)((__int64)(v4[1] - *v4) >> 2) - 1;
}
