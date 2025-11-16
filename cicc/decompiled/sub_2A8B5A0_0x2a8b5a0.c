// Function: sub_2A8B5A0
// Address: 0x2a8b5a0
//
__int64 __fastcall sub_2A8B5A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 i; // r13
  bool v6; // al
  __int64 v7; // rdi
  const void **v8; // rsi
  __int64 v9; // r15
  __int64 v10; // r12
  __int64 v11; // r13
  bool v12; // cc
  unsigned __int64 v13; // rdi
  __int64 v14; // rcx
  __int64 *v15; // r14
  __int64 v16; // rcx
  unsigned int v17; // edx
  const void *v18; // rcx
  __int64 v19; // r14
  bool v20; // al
  __int64 v21; // rdi
  __int64 v22; // r15
  unsigned __int64 v23; // rdi
  unsigned int v24; // eax
  unsigned __int64 v25; // rdi
  __int64 v27; // r8
  __int64 v28; // r15
  __int64 v29; // r14
  unsigned __int64 v30; // rdi
  unsigned int v33; // [rsp+8h] [rbp-78h]
  __int64 v34; // [rsp+10h] [rbp-70h]
  __int64 v35; // [rsp+10h] [rbp-70h]
  const void *v37; // [rsp+20h] [rbp-60h]
  __int64 v38; // [rsp+28h] [rbp-58h]
  const void *v39; // [rsp+38h] [rbp-48h] BYREF
  unsigned int v40; // [rsp+40h] [rbp-40h]

  v34 = (a3 - 1) / 2;
  if ( a2 < v34 )
  {
    for ( i = a2; ; i = v9 )
    {
      v9 = 2 * (i + 1);
      v14 = 48 * (i + 1);
      v10 = a1 + v14;
      v15 = (__int64 *)(a1 + v14 - 24);
      v8 = (const void **)(v15 + 1);
      v7 = a1 + v14 + 8;
      if ( *(_DWORD *)(a1 + v14 + 16) > 0x40u )
      {
        v38 = a1 + v14 + 8;
        v6 = sub_C43C50(v7, v8);
        v7 = v38;
        v8 = (const void **)(v15 + 1);
        if ( v6 )
          goto LABEL_13;
      }
      else if ( *(_QWORD *)(v10 + 8) == v15[1] )
      {
LABEL_13:
        if ( !sub_B445A0(*(_QWORD *)v10, *v15) )
          goto LABEL_6;
LABEL_5:
        --v9;
        v10 = a1 + 24 * v9;
        goto LABEL_6;
      }
      if ( (int)sub_C4C880(v7, (__int64)v8) < 0 )
        goto LABEL_5;
LABEL_6:
      v11 = a1 + 24 * i;
      v12 = *(_DWORD *)(v11 + 16) <= 0x40u;
      *(_QWORD *)v11 = *(_QWORD *)v10;
      if ( !v12 )
      {
        v13 = *(_QWORD *)(v11 + 8);
        if ( v13 )
          j_j___libc_free_0_0(v13);
      }
      *(_QWORD *)(v11 + 8) = *(_QWORD *)(v10 + 8);
      *(_DWORD *)(v11 + 16) = *(_DWORD *)(v10 + 16);
      *(_DWORD *)(v10 + 16) = 0;
      if ( v9 >= v34 )
        goto LABEL_16;
    }
  }
  v9 = a2;
  v10 = a1 + 24 * a2;
LABEL_16:
  if ( (a3 & 1) == 0 && (a3 - 2) / 2 == v9 )
  {
    v27 = v9 + 1;
    v12 = *(_DWORD *)(v10 + 16) <= 0x40u;
    v28 = 2 * (v9 + 1);
    v29 = a1 + 8 * (v28 + 4 * v27) - 24;
    *(_QWORD *)v10 = *(_QWORD *)v29;
    if ( !v12 )
    {
      v30 = *(_QWORD *)(v10 + 8);
      if ( v30 )
        j_j___libc_free_0_0(v30);
    }
    v9 = v28 - 1;
    *(_QWORD *)(v10 + 8) = *(_QWORD *)(v29 + 8);
    *(_DWORD *)(v10 + 16) = *(_DWORD *)(v29 + 16);
    *(_DWORD *)(v29 + 16) = 0;
    v10 = a1 + 24 * v9;
  }
  v16 = *(_QWORD *)a4;
  v17 = *(_DWORD *)(a4 + 16);
  *(_DWORD *)(a4 + 16) = 0;
  v35 = v16;
  v18 = *(const void **)(a4 + 8);
  v33 = v17;
  v40 = v17;
  v37 = v18;
  v39 = v18;
  v19 = (v9 - 1) / 2;
  if ( v9 > a2 )
  {
    while ( 1 )
    {
      v10 = a1 + 24 * v19;
      v21 = v10 + 8;
      if ( *(_DWORD *)(v10 + 16) > 0x40u )
      {
        v20 = sub_C43C50(v21, &v39);
        v21 = v10 + 8;
        if ( !v20 )
          goto LABEL_21;
      }
      else if ( v37 != *(const void **)(v10 + 8) )
      {
LABEL_21:
        if ( (int)sub_C4C880(v21, (__int64)&v39) >= 0 )
          goto LABEL_30;
        goto LABEL_22;
      }
      if ( !sub_B445A0(*(_QWORD *)v10, v35) )
      {
LABEL_30:
        v10 = a1 + 24 * v9;
        v24 = *(_DWORD *)(v10 + 16);
        goto LABEL_31;
      }
LABEL_22:
      v22 = a1 + 24 * v9;
      v12 = *(_DWORD *)(v22 + 16) <= 0x40u;
      *(_QWORD *)v22 = *(_QWORD *)v10;
      if ( !v12 )
      {
        v23 = *(_QWORD *)(v22 + 8);
        if ( v23 )
          j_j___libc_free_0_0(v23);
      }
      *(_QWORD *)(v22 + 8) = *(_QWORD *)(v10 + 8);
      *(_DWORD *)(v22 + 16) = *(_DWORD *)(v10 + 16);
      v9 = v19;
      *(_DWORD *)(v10 + 16) = 0;
      if ( a2 >= v19 )
      {
        *(_QWORD *)v10 = v35;
        goto LABEL_34;
      }
      v19 = (v19 - 1) / 2;
    }
  }
  v24 = *(_DWORD *)(v10 + 16);
LABEL_31:
  *(_QWORD *)v10 = v35;
  if ( v24 > 0x40 )
  {
    v25 = *(_QWORD *)(v10 + 8);
    if ( v25 )
      j_j___libc_free_0_0(v25);
  }
LABEL_34:
  *(_QWORD *)(v10 + 8) = v37;
  *(_DWORD *)(v10 + 16) = v33;
  return v33;
}
