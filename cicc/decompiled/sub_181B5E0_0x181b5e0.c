// Function: sub_181B5E0
// Address: 0x181b5e0
//
unsigned __int64 __fastcall sub_181B5E0(__int64 **a1, __int64 a2, double a3, double a4, double a5)
{
  __int64 v5; // rax
  __int64 v6; // r14
  unsigned __int64 result; // rax
  unsigned __int64 v8; // r12
  int v9; // r15d
  _BYTE *v10; // r14
  unsigned __int64 v11; // rax

  v5 = sub_15F2050(a2);
  v6 = sub_1632FA0(v5);
  result = (unsigned __int64)(sub_127FA20(v6, **(_QWORD **)(a2 - 48)) + 7) >> 3;
  if ( result )
  {
    v8 = result;
    v9 = 1;
    if ( byte_4FA9380 )
    {
      v9 = 1 << (*(unsigned __int16 *)(a2 + 18) >> 1) >> 1;
      if ( !v9 )
        v9 = sub_15A9FE0(v6, **(_QWORD **)(a2 - 48));
    }
    v10 = (_BYTE *)sub_1819D40(*a1, *(_QWORD *)(a2 - 48));
    if ( byte_4FA9000 )
    {
      v11 = sub_1819D40(*a1, *(_QWORD *)(a2 - 24));
      v10 = (_BYTE *)sub_181A560(*a1, v10, v11, a2);
    }
    return sub_1818700((__int64)*a1, *(_QWORD *)(a2 - 24), v8, v9, (__int64)v10, a2, a3, a4, a5);
  }
  return result;
}
