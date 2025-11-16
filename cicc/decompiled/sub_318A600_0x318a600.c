// Function: sub_318A600
// Address: 0x318a600
//
_QWORD *__fastcall sub_318A600(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rax
  _QWORD *v5; // rbx
  __int64 v6; // rdx
  __int64 v7; // rcx
  _QWORD *v8; // r14
  __int64 v9; // rbx
  __int64 v10; // r15
  __int64 v11; // rsi
  __int64 v12; // rbx
  __int64 i; // r13
  __int64 v14; // rsi
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18[7]; // [rsp+8h] [rbp-38h] BYREF

  sub_318A370(a1, *(_QWORD *)(a2 + 40));
  v3 = sub_3186770(a1, a2);
  if ( v3 )
  {
    sub_3186750(v18, a1, v3);
    if ( v18[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v18[0] + 8LL))(v18[0]);
  }
  v4 = sub_22077B0(0x20u);
  v5 = (_QWORD *)v4;
  if ( v4 )
  {
    sub_318EB10(v4, 0, a2, a1);
    *v5 = &unk_4A33610;
  }
  v18[0] = (__int64)v5;
  v8 = sub_3189570(a1, (__int64)v18);
  if ( v18[0] )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v18[0] + 8LL))(v18[0]);
  if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
  {
    sub_B2C6D0(a2, (__int64)v18, v6, v7);
    v9 = *(_QWORD *)(a2 + 96);
    v10 = v9 + 40LL * *(_QWORD *)(a2 + 104);
    if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
    {
      sub_B2C6D0(a2, (__int64)v18, v16, v17);
      v9 = *(_QWORD *)(a2 + 96);
    }
  }
  else
  {
    v9 = *(_QWORD *)(a2 + 96);
    v10 = v9 + 40LL * *(_QWORD *)(a2 + 104);
  }
  while ( v10 != v9 )
  {
    v11 = v9;
    v9 += 40;
    sub_31892D0(a1, v11);
  }
  v12 = *(_QWORD *)(a2 + 80);
  for ( i = a2 + 72; i != v12; v12 = *(_QWORD *)(v12 + 8) )
  {
    v14 = v12 - 24;
    if ( !v12 )
      v14 = 0;
    sub_3189900(a1, v14);
  }
  return v8;
}
