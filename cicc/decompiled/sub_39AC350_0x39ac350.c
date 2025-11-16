// Function: sub_39AC350
// Address: 0x39ac350
//
void __fastcall sub_39AC350(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rax
  __int64 v3; // r13
  __int64 v4; // rbx
  __int64 v5; // r13
  __int64 v7; // r12
  __int64 *v8; // rsi
  unsigned __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // r12
  __int64 *v12; // r13
  __int64 *v13; // rbx
  __int64 v14; // rsi
  void (__fastcall *v15)(__int64, __int64); // r15
  __int64 v16; // rax
  __int64 v17; // [rsp+8h] [rbp-58h] BYREF
  __int64 *v18; // [rsp+10h] [rbp-50h] BYREF
  __int64 *v19; // [rsp+18h] [rbp-48h]
  __int64 *v20; // [rsp+20h] [rbp-40h]

  v1 = *(_QWORD *)(a1 + 8);
  v18 = 0;
  v19 = 0;
  v2 = *(_QWORD *)(v1 + 272);
  v20 = 0;
  v3 = *(_QWORD *)(v2 + 1688);
  v4 = *(_QWORD *)(v3 + 32);
  v5 = v3 + 24;
  if ( v4 == v5 )
    return;
  do
  {
    while ( 1 )
    {
      v7 = v4 - 56;
      if ( !v4 )
        v7 = 0;
      if ( !(unsigned __int8)sub_15E3650(v7, 0) )
        goto LABEL_3;
      v17 = v7;
      v8 = v19;
      if ( v19 != v20 )
        break;
      sub_39AC1C0((__int64)&v18, v19, &v17);
LABEL_3:
      v4 = *(_QWORD *)(v4 + 8);
      if ( v5 == v4 )
        goto LABEL_11;
    }
    if ( v19 )
    {
      *v19 = v7;
      v8 = v19;
    }
    v19 = v8 + 1;
    v4 = *(_QWORD *)(v4 + 8);
  }
  while ( v5 != v4 );
LABEL_11:
  v9 = (unsigned __int64)v18;
  if ( v19 == v18 )
  {
    if ( v18 )
      goto LABEL_17;
  }
  else
  {
    v10 = *(_QWORD *)(a1 + 8);
    v11 = *(_QWORD *)(v10 + 256);
    (*(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v11 + 160LL))(
      v11,
      *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v10 + 248) + 32LL) + 664LL),
      0);
    v9 = (unsigned __int64)v18;
    v12 = v19;
    if ( v19 != v18 )
    {
      v13 = v18;
      do
      {
        v14 = *v13++;
        v15 = *(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v11 + 312LL);
        v16 = sub_396EAF0(*(_QWORD *)(a1 + 8), v14);
        v15(v11, v16);
      }
      while ( v12 != v13 );
      v9 = (unsigned __int64)v18;
    }
    if ( v9 )
LABEL_17:
      j_j___libc_free_0(v9);
  }
}
