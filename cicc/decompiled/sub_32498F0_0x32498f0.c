// Function: sub_32498F0
// Address: 0x32498f0
//
__int64 __fastcall sub_32498F0(__int64 *a1, unsigned __int64 **a2, __int16 a3, int a4, __int64 a5)
{
  __int16 v6; // r12
  __int64 result; // rax
  unsigned int v8; // r15d
  __int16 v9; // [rsp+4h] [rbp-4Ch]
  __int64 v10[8]; // [rsp+10h] [rbp-40h] BYREF

  v6 = a4;
  if ( (a4 & 0xFF0000) == 0 )
  {
    v6 = 11;
    if ( a5 != (char)a5 )
      v6 = (a5 != (int)a5) + 6;
  }
  if ( !a3
    || (*(_BYTE *)(*(_QWORD *)(a1[23] + 200) + 904LL) & 0x40) == 0
    || (v9 = a3, v8 = (unsigned __int16)sub_3220AA0(a1[26]),
                 result = sub_E06A90(v9),
                 a3 = v9,
                 v8 >= (unsigned int)result) )
  {
    WORD2(v10[0]) = a3;
    LODWORD(v10[0]) = 1;
    HIWORD(v10[0]) = v6;
    v10[1] = a5;
    return sub_3248F80(a2, a1 + 11, v10);
  }
  return result;
}
