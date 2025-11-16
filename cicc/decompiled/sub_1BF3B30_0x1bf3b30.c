// Function: sub_1BF3B30
// Address: 0x1bf3b30
//
__int64 __fastcall sub_1BF3B30(__int64 *a1, __int64 a2, unsigned __int8 a3)
{
  unsigned int v3; // ebx
  __int64 *v4; // r13
  __int64 v5; // rax
  unsigned int v6; // r13d
  _QWORD *v7; // r14
  _QWORD *v8; // r15
  __int64 v10; // rax
  __int64 v11; // r13
  char v12; // [rsp+Fh] [rbp-31h]

  v3 = a3;
  v4 = (__int64 *)a1[7];
  v5 = sub_15E0530(*v4);
  if ( sub_1602790(v5)
    || (v10 = sub_15E0530(*v4),
        v11 = sub_16033E0(v10),
        (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v11 + 32LL))(
          v11,
          "loop-vectorize",
          14))
    || (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v11 + 40LL))(
         v11,
         "loop-vectorize",
         14)
    || (v12 = (*(__int64 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v11 + 24LL))(
                v11,
                "loop-vectorize",
                14)) != 0 )
  {
    v12 = 1;
    v6 = sub_1BF3490(a1, a2);
  }
  else
  {
    v6 = sub_1BF3490(a1, a2);
    if ( !(_BYTE)v6 )
      return 0;
  }
  v7 = *(_QWORD **)(a2 + 16);
  if ( v7 != *(_QWORD **)(a2 + 8) )
  {
    v8 = *(_QWORD **)(a2 + 8);
    do
    {
      if ( !(unsigned __int8)sub_1BF3B30(a1, *v8, v3) )
      {
        if ( !v12 )
          return 0;
        v6 = 0;
      }
      ++v8;
    }
    while ( v7 != v8 );
  }
  return v6;
}
