// Function: sub_324AB90
// Address: 0x324ab90
//
__int64 __fastcall sub_324AB90(__int64 *a1, __int64 a2, __int16 a3, __int64 a4, __int64 a5)
{
  _QWORD *v7; // rax
  __int64 v8; // rbx
  __int16 v9; // r14
  __int64 result; // rax
  unsigned int v11; // r15d
  __int64 v13[8]; // [rsp+10h] [rbp-40h] BYREF

  v7 = (_QWORD *)sub_A777F0(0x10u, a1 + 11);
  v8 = (__int64)v7;
  if ( v7 )
  {
    *v7 = a4;
    v7[1] = a5;
  }
  v9 = sub_3222A40(a1[26]);
  if ( !a3
    || (*(_BYTE *)(*(_QWORD *)(a1[23] + 200) + 904LL) & 0x40) == 0
    || (v11 = (unsigned __int16)sub_3220AA0(a1[26]), result = sub_E06A90(a3), v11 >= (unsigned int)result) )
  {
    LODWORD(v13[0]) = 6;
    WORD2(v13[0]) = a3;
    HIWORD(v13[0]) = v9;
    v13[1] = v8;
    return sub_3248F80((unsigned __int64 **)(a2 + 8), a1 + 11, v13);
  }
  return result;
}
