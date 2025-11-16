// Function: sub_38AB2F0
// Address: 0x38ab2f0
//
__int64 __fastcall sub_38AB2F0(
        __int64 a1,
        _QWORD *a2,
        unsigned __int64 *a3,
        __int64 *a4,
        double a5,
        double a6,
        double a7)
{
  __int64 result; // rax
  unsigned __int64 v9; // rsi
  __int64 v10; // [rsp+8h] [rbp-48h] BYREF
  const char *v11; // [rsp+10h] [rbp-40h] BYREF
  char v12; // [rsp+20h] [rbp-30h]
  char v13; // [rsp+21h] [rbp-2Fh]

  *a3 = *(_QWORD *)(a1 + 56);
  result = sub_38AB270((__int64 **)a1, &v10, a4, a5, a6, a7);
  if ( !(_BYTE)result )
  {
    if ( *(_BYTE *)(v10 + 16) == 18 )
    {
      *a2 = v10;
    }
    else
    {
      v9 = *a3;
      v13 = 1;
      v12 = 3;
      v11 = "expected a basic block";
      return sub_38814C0(a1 + 8, v9, (__int64)&v11);
    }
  }
  return result;
}
