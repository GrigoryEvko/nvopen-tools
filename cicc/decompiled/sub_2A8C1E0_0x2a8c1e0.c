// Function: sub_2A8C1E0
// Address: 0x2a8c1e0
//
__int64 __fastcall sub_2A8C1E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rbx
  __int64 i; // r13
  __int64 v8; // r15
  bool v9; // al
  __int64 v10; // rcx
  __int64 v11; // r13
  bool v12; // cc
  unsigned __int64 v13; // rdi
  unsigned int v14; // eax
  __int64 v15; // rbx
  __int64 v16; // r13
  unsigned __int64 v17; // rdi
  __int64 v18; // r14
  bool v19; // al
  unsigned int v20; // eax
  unsigned __int64 v21; // rdi
  unsigned int v23; // eax
  __int64 v24; // r8
  __int64 v25; // r15
  __int64 v26; // rbx
  unsigned __int64 v27; // rdi
  unsigned int v28; // eax
  unsigned int v30; // [rsp+0h] [rbp-50h]
  __int64 v31; // [rsp+8h] [rbp-48h]
  __int64 v32; // [rsp+8h] [rbp-48h]
  __int64 v33; // [rsp+18h] [rbp-38h]
  __int64 v34; // [rsp+18h] [rbp-38h]
  __int64 v35; // [rsp+18h] [rbp-38h]

  v6 = (a3 - 1) / 2;
  v31 = a3 & 1;
  if ( v6 <= a2 )
  {
    v10 = a1 + 24 * a2;
    if ( (a3 & 1) != 0 )
    {
      v28 = *(_DWORD *)(a4 + 16);
      v15 = *(_QWORD *)a4;
      *(_DWORD *)(a4 + 16) = 0;
      v30 = v28;
      v32 = *(_QWORD *)(a4 + 8);
      v20 = *(_DWORD *)(v10 + 16);
      goto LABEL_21;
    }
    v8 = a2;
  }
  else
  {
    for ( i = a2; ; i = v8 )
    {
      v8 = 2 * (i + 1);
      v9 = sub_B445A0(*(_QWORD *)(a1 + 48 * (i + 1)), *(_QWORD *)(a1 + 48 * (i + 1) - 24));
      v10 = a1 + 48 * (i + 1);
      if ( v9 )
      {
        --v8;
        v10 = a1 + 24 * v8;
      }
      v11 = a1 + 24 * i;
      v12 = *(_DWORD *)(v11 + 16) <= 0x40u;
      *(_QWORD *)v11 = *(_QWORD *)v10;
      if ( !v12 )
      {
        v13 = *(_QWORD *)(v11 + 8);
        if ( v13 )
        {
          v33 = v10;
          j_j___libc_free_0_0(v13);
          v10 = v33;
        }
      }
      *(_QWORD *)(v11 + 8) = *(_QWORD *)(v10 + 8);
      *(_DWORD *)(v11 + 16) = *(_DWORD *)(v10 + 16);
      *(_DWORD *)(v10 + 16) = 0;
      if ( v8 >= v6 )
        break;
    }
    if ( v31 )
    {
      v23 = *(_DWORD *)(a4 + 16);
      v15 = *(_QWORD *)a4;
      *(_DWORD *)(a4 + 16) = 0;
      v30 = v23;
      v32 = *(_QWORD *)(a4 + 8);
      v16 = (v8 - 1) / 2;
      goto LABEL_19;
    }
  }
  if ( (a3 - 2) / 2 == v8 )
  {
    v24 = v8 + 1;
    v12 = *(_DWORD *)(v10 + 16) <= 0x40u;
    v25 = 2 * (v8 + 1);
    v26 = a1 + 8 * (v25 + 4 * v24) - 24;
    *(_QWORD *)v10 = *(_QWORD *)v26;
    if ( !v12 )
    {
      v27 = *(_QWORD *)(v10 + 8);
      if ( v27 )
      {
        v35 = v10;
        j_j___libc_free_0_0(v27);
        v10 = v35;
      }
    }
    v8 = v25 - 1;
    *(_QWORD *)(v10 + 8) = *(_QWORD *)(v26 + 8);
    *(_DWORD *)(v10 + 16) = *(_DWORD *)(v26 + 16);
    *(_DWORD *)(v26 + 16) = 0;
    v10 = a1 + 24 * v8;
  }
  v14 = *(_DWORD *)(a4 + 16);
  v15 = *(_QWORD *)a4;
  *(_DWORD *)(a4 + 16) = 0;
  v30 = v14;
  v32 = *(_QWORD *)(a4 + 8);
  v16 = (v8 - 1) / 2;
  if ( v8 > a2 )
  {
LABEL_19:
    while ( 1 )
    {
      v18 = a1 + 24 * v16;
      v19 = sub_B445A0(*(_QWORD *)v18, v15);
      v10 = a1 + 24 * v8;
      if ( !v19 )
        break;
      v12 = *(_DWORD *)(v10 + 16) <= 0x40u;
      *(_QWORD *)v10 = *(_QWORD *)v18;
      if ( !v12 )
      {
        v17 = *(_QWORD *)(v10 + 8);
        if ( v17 )
        {
          j_j___libc_free_0_0(v17);
          v10 = a1 + 24 * v8;
        }
      }
      v8 = v16;
      *(_QWORD *)(v10 + 8) = *(_QWORD *)(v18 + 8);
      *(_DWORD *)(v10 + 16) = *(_DWORD *)(v18 + 16);
      *(_DWORD *)(v18 + 16) = 0;
      if ( a2 >= v16 )
      {
        *(_QWORD *)v18 = v15;
        v10 = a1 + 24 * v16;
        goto LABEL_24;
      }
      v16 = (v16 - 1) / 2;
    }
  }
  v20 = *(_DWORD *)(v10 + 16);
LABEL_21:
  *(_QWORD *)v10 = v15;
  if ( v20 > 0x40 )
  {
    v21 = *(_QWORD *)(v10 + 8);
    if ( v21 )
    {
      v34 = v10;
      j_j___libc_free_0_0(v21);
      v10 = v34;
    }
  }
LABEL_24:
  *(_QWORD *)(v10 + 8) = v32;
  *(_DWORD *)(v10 + 16) = v30;
  return v30;
}
