// Function: sub_EDB300
// Address: 0xedb300
//
__int64 *__fastcall sub_EDB300(__int64 *a1, __int64 *a2, __int64 *a3)
{
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v6; // rdx
  __int64 *v7; // r14
  bool v8; // cf
  unsigned __int64 v9; // rax
  __int64 v10; // r10
  __int64 v11; // r15
  bool v12; // zf
  __int64 v13; // r10
  __int64 v14; // r13
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 *v17; // rax
  __int64 v18; // rdx
  _QWORD *v19; // r11
  __int64 v20; // rdx
  __int64 v21; // r13
  __int64 i; // r15
  __int64 *v23; // rax
  __int64 v24; // rdx
  _QWORD *v25; // r10
  __int64 v26; // rax
  __int64 v27; // r13
  __int64 *v28; // rax
  __int64 *v29; // r12
  _QWORD *v30; // r13
  int v31; // eax
  __int64 j; // r12
  _QWORD *v33; // r13
  __int64 v35; // r15
  __int64 v36; // rax
  __int64 *v37; // [rsp+0h] [rbp-60h]
  __int64 *v38; // [rsp+0h] [rbp-60h]
  __int64 *v39; // [rsp+0h] [rbp-60h]
  __int64 v40; // [rsp+8h] [rbp-58h]
  __int64 v41; // [rsp+18h] [rbp-48h]
  __int64 v42; // [rsp+20h] [rbp-40h]
  __int64 v43; // [rsp+28h] [rbp-38h]
  __int64 *v44; // [rsp+28h] [rbp-38h]
  _QWORD *v45; // [rsp+28h] [rbp-38h]
  __int64 *v46; // [rsp+28h] [rbp-38h]
  __int64 v47; // [rsp+28h] [rbp-38h]
  __int64 *v48; // [rsp+28h] [rbp-38h]
  _QWORD *v49; // [rsp+28h] [rbp-38h]

  v3 = a1[1];
  v42 = *a1;
  v4 = (v3 - *a1) >> 5;
  if ( v4 == 0x3FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v6 = 1;
  if ( v4 )
    v6 = (v3 - *a1) >> 5;
  v7 = a2;
  v8 = __CFADD__(v6, v4);
  v9 = v6 + v4;
  v10 = (__int64)a2 - v42;
  if ( v8 )
  {
    v35 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v9 )
    {
      v40 = 0;
      v11 = 32;
      v41 = 0;
      goto LABEL_7;
    }
    if ( v9 > 0x3FFFFFFFFFFFFFFLL )
      v9 = 0x3FFFFFFFFFFFFFFLL;
    v35 = 32 * v9;
  }
  v39 = a3;
  v36 = sub_22077B0(v35);
  v10 = (__int64)a2 - v42;
  a3 = v39;
  v41 = v36;
  v40 = v36 + v35;
  v11 = v36 + 32;
LABEL_7:
  v12 = v41 + v10 == 0;
  v13 = v41 + v10;
  v14 = v13;
  if ( !v12 )
  {
    v15 = a3[1];
    v16 = *a3;
    *(_QWORD *)(v13 + 8) = 0;
    *(_QWORD *)(v13 + 16) = 0;
    *(_BYTE *)(v13 + 24) = 0;
    *(_QWORD *)v13 = v16;
    v43 = v15;
    if ( v15 )
    {
      v37 = a3;
      v17 = (__int64 *)sub_22077B0(32);
      a3 = v37;
      if ( v17 )
      {
        v18 = v43;
        v44 = v17;
        *v17 = (__int64)(v17 + 2);
        sub_ED71E0(v17, *(_BYTE **)v18, *(_QWORD *)v18 + *(_QWORD *)(v18 + 8));
        a3 = v37;
        v17 = v44;
      }
      v19 = *(_QWORD **)(v14 + 8);
      *(_QWORD *)(v14 + 8) = v17;
      if ( v19 )
      {
        if ( (_QWORD *)*v19 != v19 + 2 )
        {
          v38 = a3;
          v45 = v19;
          j_j___libc_free_0(*v19, v19[2] + 1LL);
          a3 = v38;
          v19 = v45;
        }
        v46 = a3;
        j_j___libc_free_0(v19, 32);
        a3 = v46;
      }
    }
    v20 = a3[2];
    *(_BYTE *)(v14 + 24) = *((_BYTE *)a3 + 24);
    *(_QWORD *)(v14 + 16) = v20;
  }
  v21 = v42;
  if ( a2 != (__int64 *)v42 )
  {
    for ( i = v41; ; i += 32 )
    {
      if ( i )
      {
        *(_QWORD *)(i + 8) = 0;
        *(_DWORD *)(i + 16) = 0;
        *(_DWORD *)(i + 20) = 0;
        *(_BYTE *)(i + 24) = 0;
        *(_QWORD *)i = *(_QWORD *)v21;
        v47 = *(_QWORD *)(v21 + 8);
        if ( v47 )
        {
          v23 = (__int64 *)sub_22077B0(32);
          if ( v23 )
          {
            v24 = v47;
            v48 = v23;
            *v23 = (__int64)(v23 + 2);
            sub_ED71E0(v23, *(_BYTE **)v24, *(_QWORD *)v24 + *(_QWORD *)(v24 + 8));
            v23 = v48;
          }
          v25 = *(_QWORD **)(i + 8);
          *(_QWORD *)(i + 8) = v23;
          if ( v25 )
          {
            if ( (_QWORD *)*v25 != v25 + 2 )
            {
              v49 = v25;
              j_j___libc_free_0(*v25, v25[2] + 1LL);
              v25 = v49;
            }
            j_j___libc_free_0(v25, 32);
          }
        }
        *(_DWORD *)(i + 16) = *(_DWORD *)(v21 + 16);
        *(_DWORD *)(i + 20) = *(_DWORD *)(v21 + 20);
        *(_BYTE *)(i + 24) = *(_BYTE *)(v21 + 24);
      }
      v21 += 32;
      if ( a2 == (__int64 *)v21 )
        break;
    }
    v11 = i + 64;
  }
  if ( a2 != (__int64 *)v3 )
  {
    do
    {
      v26 = *v7;
      v27 = v7[1];
      *(_QWORD *)(v11 + 8) = 0;
      *(_DWORD *)(v11 + 16) = 0;
      *(_DWORD *)(v11 + 20) = 0;
      *(_BYTE *)(v11 + 24) = 0;
      *(_QWORD *)v11 = v26;
      if ( v27 )
      {
        v28 = (__int64 *)sub_22077B0(32);
        v29 = v28;
        if ( v28 )
        {
          *v28 = (__int64)(v28 + 2);
          sub_ED71E0(v28, *(_BYTE **)v27, *(_QWORD *)v27 + *(_QWORD *)(v27 + 8));
        }
        v30 = *(_QWORD **)(v11 + 8);
        *(_QWORD *)(v11 + 8) = v29;
        if ( v30 )
        {
          if ( (_QWORD *)*v30 != v30 + 2 )
            j_j___libc_free_0(*v30, v30[2] + 1LL);
          j_j___libc_free_0(v30, 32);
        }
      }
      v31 = *((_DWORD *)v7 + 4);
      v7 += 4;
      v11 += 32;
      *(_DWORD *)(v11 - 16) = v31;
      *(_DWORD *)(v11 - 12) = *((_DWORD *)v7 - 3);
      *(_BYTE *)(v11 - 8) = *((_BYTE *)v7 - 8);
    }
    while ( (__int64 *)v3 != v7 );
  }
  for ( j = v42; v3 != j; j += 32 )
  {
    v33 = *(_QWORD **)(j + 8);
    if ( v33 )
    {
      if ( (_QWORD *)*v33 != v33 + 2 )
        j_j___libc_free_0(*v33, v33[2] + 1LL);
      j_j___libc_free_0(v33, 32);
    }
  }
  if ( v42 )
    j_j___libc_free_0(v42, a1[2] - v42);
  *a1 = v41;
  a1[1] = v11;
  a1[2] = v40;
  return a1;
}
