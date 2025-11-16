// Function: sub_121B570
// Address: 0x121b570
//
__int64 __fastcall sub_121B570(__int64 a1, __int64 *a2)
{
  __int64 v4; // r13
  unsigned int v5; // r15d
  __int64 v6; // rax
  __int64 v7; // r9
  __int64 v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r10
  __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rax
  unsigned int v16; // r8d
  __int64 v17; // rax
  __int64 v18; // rdx
  _BOOL8 v19; // rdi
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r10
  __int64 v25; // r9
  __int64 v26; // r8
  __int64 v27; // rdi
  __int64 v28; // rax
  __int64 v29; // r13
  __int64 v30; // rcx
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rsi
  unsigned int v34; // [rsp+Ch] [rbp-64h]
  __int64 v35; // [rsp+10h] [rbp-60h]
  __int64 v36; // [rsp+20h] [rbp-50h]
  __int64 v37; // [rsp+20h] [rbp-50h]
  __int64 v38; // [rsp+20h] [rbp-50h]
  __int64 v39; // [rsp+28h] [rbp-48h]
  __int64 v40; // [rsp+28h] [rbp-48h]
  __int64 v41; // [rsp+28h] [rbp-48h]
  unsigned int v42; // [rsp+34h] [rbp-3Ch] BYREF
  unsigned int *v43; // [rsp+38h] [rbp-38h] BYREF

  v42 = 0;
  v4 = *(_QWORD *)(a1 + 232);
  v5 = sub_120BD00(a1, &v42);
  if ( (_BYTE)v5 )
    return v5;
  v6 = *(_QWORD *)(a1 + 1016);
  v7 = a1 + 1008;
  if ( v6 )
  {
    v8 = a1 + 1008;
    do
    {
      while ( 1 )
      {
        v9 = *(_QWORD *)(v6 + 16);
        v10 = *(_QWORD *)(v6 + 24);
        if ( *(_DWORD *)(v6 + 32) >= v42 )
          break;
        v6 = *(_QWORD *)(v6 + 24);
        if ( !v10 )
          goto LABEL_7;
      }
      v8 = v6;
      v6 = *(_QWORD *)(v6 + 16);
    }
    while ( v9 );
LABEL_7:
    if ( v7 != v8 && v42 >= *(_DWORD *)(v8 + 32) )
    {
      *a2 = *(_QWORD *)(v8 + 40);
      return v5;
    }
  }
  v11 = *(_QWORD *)(a1 + 1064);
  if ( !v11 )
  {
    v12 = a1 + 1056;
LABEL_16:
    v36 = v12;
    v35 = a1 + 1056;
    v15 = sub_22077B0(56);
    v16 = v42;
    *(_QWORD *)(v15 + 40) = 0;
    *(_DWORD *)(v15 + 32) = v16;
    *(_QWORD *)(v15 + 48) = 0;
    v34 = v16;
    v39 = v15;
    v17 = sub_121B220((_QWORD *)(a1 + 1048), v36, (unsigned int *)(v15 + 32));
    if ( v18 )
    {
      v19 = v17 || v35 == v18 || v34 < *(_DWORD *)(v18 + 32);
      sub_220F040(v19, v39, v18, v35);
      v12 = v39;
      v7 = a1 + 1008;
      ++*(_QWORD *)(a1 + 1088);
    }
    else
    {
      v38 = v17;
      j_j___libc_free_0(v39, 56);
      v7 = a1 + 1008;
      v12 = v38;
    }
    goto LABEL_24;
  }
  v12 = a1 + 1056;
  do
  {
    while ( 1 )
    {
      v13 = *(_QWORD *)(v11 + 16);
      v14 = *(_QWORD *)(v11 + 24);
      if ( *(_DWORD *)(v11 + 32) >= v42 )
        break;
      v11 = *(_QWORD *)(v11 + 24);
      if ( !v14 )
        goto LABEL_14;
    }
    v12 = v11;
    v11 = *(_QWORD *)(v11 + 16);
  }
  while ( v13 );
LABEL_14:
  if ( v12 == a1 + 1056 || v42 < *(_DWORD *)(v12 + 32) )
    goto LABEL_16;
LABEL_24:
  v37 = v12;
  v40 = v7;
  v21 = sub_B9C770(*(__int64 **)a1, 0, 0, 2, 1);
  v24 = v37;
  v25 = v40;
  v26 = v21;
  v27 = *(_QWORD *)(v37 + 40);
  *(_QWORD *)(v37 + 40) = v21;
  if ( v27 )
  {
    sub_BA65D0(v27, 0, v22, v23, v21);
    v24 = v37;
    v25 = v40;
    v26 = *(_QWORD *)(v37 + 40);
  }
  *(_QWORD *)(v24 + 48) = v4;
  *a2 = v26;
  v28 = *(_QWORD *)(a1 + 1016);
  if ( v28 )
  {
    v29 = v25;
    do
    {
      while ( 1 )
      {
        v30 = *(_QWORD *)(v28 + 16);
        v31 = *(_QWORD *)(v28 + 24);
        if ( *(_DWORD *)(v28 + 32) >= v42 )
          break;
        v28 = *(_QWORD *)(v28 + 24);
        if ( !v31 )
          goto LABEL_31;
      }
      v29 = v28;
      v28 = *(_QWORD *)(v28 + 16);
    }
    while ( v30 );
LABEL_31:
    if ( v25 != v29 && v42 >= *(_DWORD *)(v29 + 32) )
      goto LABEL_34;
  }
  else
  {
    v29 = v25;
  }
  v43 = &v42;
  v32 = sub_121B4C0((_QWORD *)(a1 + 1000), v29, &v43);
  v26 = *a2;
  v29 = v32;
LABEL_34:
  v33 = *(_QWORD *)(v29 + 40);
  if ( v33 )
  {
    v41 = v26;
    sub_B91220(v29 + 40, v33);
    v26 = v41;
  }
  *(_QWORD *)(v29 + 40) = v26;
  if ( v26 )
    sub_B96E90(v29 + 40, v26, 1);
  return v5;
}
