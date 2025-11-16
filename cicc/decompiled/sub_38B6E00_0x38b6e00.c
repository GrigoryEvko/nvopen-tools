// Function: sub_38B6E00
// Address: 0x38b6e00
//
__int64 __fastcall sub_38B6E00(__int64 a1)
{
  unsigned int v1; // r13d
  __int64 result; // rax
  int v3; // eax
  unsigned __int64 v4; // rsi
  const char *v5; // [rsp+0h] [rbp-40h] BYREF
  char v6; // [rsp+10h] [rbp-30h]
  char v7; // [rsp+11h] [rbp-2Fh]

  *(_BYTE *)(a1 + 168) = 1;
  v1 = *(_DWORD *)(a1 + 104);
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  result = sub_388AF10(a1, 3, "expected '=' here");
  if ( !(_BYTE)result )
  {
    if ( *(_QWORD *)(a1 + 184) )
    {
      v3 = *(_DWORD *)(a1 + 64);
      switch ( v3 )
      {
        case 305:
          return sub_38B6A00(a1, v1);
        case 340:
          return sub_38B39D0(a1, v1);
        case 93:
          return sub_38B2790(a1, v1);
        default:
          v4 = *(_QWORD *)(a1 + 56);
          v7 = 1;
          v6 = 3;
          v5 = "unexpected summary kind";
          return sub_38814C0(a1 + 8, v4, (__int64)&v5);
      }
    }
    else
    {
      return sub_388AF70(a1);
    }
  }
  return result;
}
