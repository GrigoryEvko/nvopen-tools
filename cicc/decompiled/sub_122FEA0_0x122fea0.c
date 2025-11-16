// Function: sub_122FEA0
// Address: 0x122fea0
//
__int64 __fastcall sub_122FEA0(__int64 a1, _QWORD *a2, unsigned __int64 *a3, __int64 *a4)
{
  __int64 result; // rax
  unsigned __int64 v6; // rsi
  _BYTE *v7; // [rsp+8h] [rbp-58h] BYREF
  const char *v8; // [rsp+10h] [rbp-50h] BYREF
  char v9; // [rsp+30h] [rbp-30h]
  char v10; // [rsp+31h] [rbp-2Fh]

  *a3 = *(_QWORD *)(a1 + 232);
  result = sub_122FE20((__int64 **)a1, (__int64 *)&v7, a4);
  if ( !(_BYTE)result )
  {
    if ( *v7 == 23 )
    {
      *a2 = v7;
    }
    else
    {
      v6 = *a3;
      v10 = 1;
      v8 = "expected a basic block";
      v9 = 3;
      sub_11FD800(a1 + 176, v6, (__int64)&v8, 1);
      return 1;
    }
  }
  return result;
}
