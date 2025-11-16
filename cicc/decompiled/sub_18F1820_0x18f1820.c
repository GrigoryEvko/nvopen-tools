// Function: sub_18F1820
// Address: 0x18f1820
//
__int64 __fastcall sub_18F1820(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // r14
  __int64 v6; // r13
  __int64 v7; // r12
  _QWORD *v8; // r12
  unsigned __int8 v9; // r15
  unsigned __int8 v10; // [rsp+Fh] [rbp-31h]

  v10 = sub_16368E0(a1, a2);
  if ( v10 )
    return 0;
  v3 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4F9B6E8, 1u);
  if ( v3 && (v4 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v3 + 104LL))(v3, &unk_4F9B6E8)) != 0 )
    v5 = v4 + 360;
  else
    v5 = 0;
  v6 = *(_QWORD *)(a2 + 48);
  if ( v6 == a2 + 40 )
  {
    return 0;
  }
  else
  {
    do
    {
      v7 = v6;
      v6 = *(_QWORD *)(v6 + 8);
      v8 = (_QWORD *)(v7 - 24);
      v9 = sub_1AE9990(v8, v5);
      if ( v9 )
      {
        sub_1AEAA40(v8);
        sub_15F20C0(v8);
        v10 = v9;
      }
    }
    while ( v6 != a2 + 40 );
  }
  return v10;
}
