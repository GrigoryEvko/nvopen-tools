// Function: sub_31A6EB0
// Address: 0x31a6eb0
//
__int64 __fastcall sub_31A6EB0(__int64 a1, __int64 a2, unsigned __int8 a3)
{
  unsigned int v3; // ebx
  __int64 v4; // r13
  unsigned int v5; // r13d
  _QWORD *v6; // r14
  _QWORD *v7; // r15
  __int64 v9; // r13
  char v10; // [rsp+Fh] [rbp-31h]

  v3 = a3;
  v4 = sub_B2BE50(**(_QWORD **)(a1 + 64));
  if ( sub_B6EA50(v4)
    || (v9 = sub_B6F970(v4),
        (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v9 + 32LL))(
          v9,
          "loop-vectorize",
          14))
    || (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v9 + 40LL))(
         v9,
         "loop-vectorize",
         14)
    || (v10 = (*(__int64 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v9 + 24LL))(
                v9,
                "loop-vectorize",
                14)) != 0 )
  {
    v10 = 1;
    v5 = sub_31A6C80(a1, a2);
  }
  else
  {
    v5 = sub_31A6C80(a1, a2);
    if ( !(_BYTE)v5 )
      return 0;
  }
  v6 = *(_QWORD **)(a2 + 16);
  if ( v6 != *(_QWORD **)(a2 + 8) )
  {
    v7 = *(_QWORD **)(a2 + 8);
    do
    {
      if ( !(unsigned __int8)sub_31A6EB0(a1, *v7, v3) )
      {
        if ( !v10 )
          return 0;
        v5 = 0;
      }
      ++v7;
    }
    while ( v6 != v7 );
  }
  return v5;
}
