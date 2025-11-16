// Function: sub_3249320
// Address: 0x3249320
//
__int64 __fastcall sub_3249320(__int64 *a1, unsigned __int64 **a2, __int16 a3, __int16 a4, __int64 a5)
{
  __int64 result; // rax
  unsigned int v8; // r15d
  __int16 v9; // [rsp+Ch] [rbp-44h]
  __int64 v10[8]; // [rsp+10h] [rbp-40h] BYREF

  if ( !a3
    || (*(_BYTE *)(*(_QWORD *)(a1[23] + 200) + 904LL) & 0x40) == 0
    || (v9 = a4, v8 = (unsigned __int16)sub_3220AA0(a1[26]),
                 result = sub_E06A90(a3),
                 a4 = v9,
                 v8 >= (unsigned int)result) )
  {
    WORD2(v10[0]) = a3;
    LODWORD(v10[0]) = 4;
    HIWORD(v10[0]) = a4;
    v10[1] = a5;
    return sub_3248F80(a2, a1 + 11, v10);
  }
  return result;
}
