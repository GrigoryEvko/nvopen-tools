// Function: sub_D87410
// Address: 0xd87410
//
__int64 __fastcall sub_D87410(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v7; // r13
  __int64 v8; // r12
  bool v9; // cc
  __int64 v10; // rdi
  __int64 v11; // rdi
  __int64 *v12; // rdx
  unsigned __int64 v13; // r10
  __int64 v14; // rcx
  unsigned __int64 *v15; // r15
  int v16; // esi
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // r14
  unsigned int v19; // esi
  __int64 v20; // r13
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rdi
  __int64 v24; // r12
  __int64 v25; // rdi
  __int64 v26; // rdi
  __int64 *v27; // rdi
  _QWORD *v28; // r12
  unsigned __int64 v29; // [rsp+8h] [rbp-68h]
  __int64 v30; // [rsp+10h] [rbp-60h]
  __int64 v31; // [rsp+20h] [rbp-50h]
  __int64 *v32; // [rsp+28h] [rbp-48h]
  __int64 *v33; // [rsp+28h] [rbp-48h]
  int v34; // [rsp+28h] [rbp-48h]
  __int64 *v35; // [rsp+28h] [rbp-48h]
  __int64 v36; // [rsp+30h] [rbp-40h]
  __int64 v37; // [rsp+30h] [rbp-40h]
  _QWORD *v38; // [rsp+30h] [rbp-40h]
  __int64 v39; // [rsp+30h] [rbp-40h]
  __int64 v40; // [rsp+38h] [rbp-38h]

  v7 = (a3 - 1) / 2;
  v40 = a2;
  if ( a2 < v7 )
  {
    while ( 1 )
    {
      v14 = 2 * (a2 + 1) - 1;
      v12 = (__int64 *)(a1 + 96 * (a2 + 1));
      v15 = (unsigned __int64 *)(a1 + 48 * v14);
      v13 = *v15;
      if ( *v12 >= *v15 )
      {
        if ( *v12 == v13 )
        {
          if ( *(_QWORD *)(v12[1] & 0xFFFFFFFFFFFFFFF8LL) >= *(_QWORD *)(v15[1] & 0xFFFFFFFFFFFFFFF8LL) )
          {
            v15 = (unsigned __int64 *)(a1 + 96 * (a2 + 1));
            v14 = 2 * (a2 + 1);
          }
        }
        else
        {
          v13 = *v12;
          v15 = (unsigned __int64 *)(a1 + 96 * (a2 + 1));
          v14 = 2 * (a2 + 1);
        }
      }
      v8 = a1 + 48 * a2;
      *(_QWORD *)v8 = v13;
      v9 = *(_DWORD *)(v8 + 24) <= 0x40u;
      *(_QWORD *)(v8 + 8) = v15[1];
      if ( !v9 )
      {
        v10 = *(_QWORD *)(v8 + 16);
        if ( v10 )
        {
          v32 = a4;
          v36 = v14;
          j_j___libc_free_0_0(v10);
          a4 = v32;
          v14 = v36;
        }
      }
      *(_QWORD *)(v8 + 16) = v15[2];
      *(_DWORD *)(v8 + 24) = *((_DWORD *)v15 + 6);
      *((_DWORD *)v15 + 6) = 0;
      if ( *(_DWORD *)(v8 + 40) > 0x40u )
      {
        v11 = *(_QWORD *)(v8 + 32);
        if ( v11 )
        {
          v33 = a4;
          v37 = v14;
          j_j___libc_free_0_0(v11);
          a4 = v33;
          v14 = v37;
        }
      }
      *(_QWORD *)(v8 + 32) = v15[4];
      *(_DWORD *)(v8 + 40) = *((_DWORD *)v15 + 10);
      *((_DWORD *)v15 + 10) = 0;
      if ( v14 >= v7 )
        break;
      a2 = v14;
    }
  }
  else
  {
    v14 = a2;
    v15 = (unsigned __int64 *)(a1 + 48 * a2);
  }
  if ( (a3 & 1) == 0 && (a3 - 2) / 2 == v14 )
  {
    v27 = (__int64 *)(v15 + 2);
    v35 = a4;
    v39 = 2 * v14 + 1;
    v28 = (_QWORD *)(a1 + 48 * v39);
    *v15 = *v28;
    v15[1] = v28[1];
    v15 = v28;
    sub_D859E0(v27, v28 + 2);
    a4 = v35;
    v14 = v39;
  }
  v16 = *((_DWORD *)a4 + 6);
  v17 = a4[1];
  *((_DWORD *)a4 + 6) = 0;
  v18 = *a4;
  v34 = v16;
  v29 = v17;
  v31 = a4[2];
  v19 = *((_DWORD *)a4 + 10);
  *((_DWORD *)a4 + 10) = 0;
  v30 = a4[4];
  v20 = (v14 - 1) / 2;
  if ( v14 > v40 )
  {
    v38 = (_QWORD *)(v17 & 0xFFFFFFFFFFFFFFF8LL);
    while ( 1 )
    {
      v15 = (unsigned __int64 *)(a1 + 48 * v20);
      if ( v18 <= *v15 && (v18 != *v15 || *(_QWORD *)(v15[1] & 0xFFFFFFFFFFFFFFF8LL) >= *v38) )
        break;
      v24 = a1 + 48 * v14;
      *(_QWORD *)v24 = *v15;
      v9 = *(_DWORD *)(v24 + 24) <= 0x40u;
      *(_QWORD *)(v24 + 8) = v15[1];
      if ( !v9 )
      {
        v25 = *(_QWORD *)(v24 + 16);
        if ( v25 )
          j_j___libc_free_0_0(v25);
      }
      *(_QWORD *)(v24 + 16) = v15[2];
      *(_DWORD *)(v24 + 24) = *((_DWORD *)v15 + 6);
      *((_DWORD *)v15 + 6) = 0;
      if ( *(_DWORD *)(v24 + 40) > 0x40u )
      {
        v26 = *(_QWORD *)(v24 + 32);
        if ( v26 )
          j_j___libc_free_0_0(v26);
      }
      *(_QWORD *)(v24 + 32) = v15[4];
      *(_DWORD *)(v24 + 40) = *((_DWORD *)v15 + 10);
      *((_DWORD *)v15 + 10) = 0;
      v14 = v20;
      if ( v40 >= v20 )
        goto LABEL_25;
      v20 = (v20 - 1) / 2;
    }
    v15 = (unsigned __int64 *)(a1 + 48 * v14);
  }
LABEL_25:
  v9 = *((_DWORD *)v15 + 6) <= 0x40u;
  *v15 = v18;
  v15[1] = v29;
  if ( !v9 )
  {
    v21 = v15[2];
    if ( v21 )
      j_j___libc_free_0_0(v21);
  }
  v9 = *((_DWORD *)v15 + 10) <= 0x40u;
  v15[2] = v31;
  *((_DWORD *)v15 + 6) = v34;
  if ( !v9 )
  {
    v22 = v15[4];
    if ( v22 )
      j_j___libc_free_0_0(v22);
  }
  v15[4] = v30;
  *((_DWORD *)v15 + 10) = v19;
  return v19;
}
