// Function: sub_130C1E0
// Address: 0x130c1e0
//
__int64 __fastcall sub_130C1E0(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        unsigned __int8 a5,
        unsigned __int8 a6,
        char a7)
{
  unsigned int v7; // r15d
  __int64 v9; // r14
  __int64 result; // rax
  __int64 v13; // [rsp+18h] [rbp-38h]

  v7 = a5;
  v9 = sub_131C0E0(*(_QWORD *)(a2 + 58376));
  if ( a6 == 1 && !a7 )
  {
LABEL_3:
    result = sub_130C0F0(a1, a2, v9, a3 + 0x2000, 4096, v7, 0);
    if ( result )
    {
      v13 = result;
      sub_130D840(a1, v9, result, *(_QWORD *)(a2 + 58384), 1, 1, 1);
      return v13;
    }
    return result;
  }
  result = sub_130C0F0(a1, a2, v9, a3, a4, v7, a6);
  if ( !result && a6 )
  {
    if ( unk_4C6F2C8 && a7 )
      return sub_130D630(a1, a2 + 58520, a2, v9, a3, v7);
    goto LABEL_3;
  }
  return result;
}
