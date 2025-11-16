// Function: sub_3737B50
// Address: 0x3737b50
//
__int64 __fastcall sub_3737B50(__int64 *a1, __int64 a2, __int16 a3, __int64 a4)
{
  unsigned __int64 **v5; // r14
  __int64 result; // rax
  unsigned __int16 v8; // r13
  unsigned int v9; // r13d
  __int64 v10; // [rsp+8h] [rbp-48h]
  __int64 v11; // [rsp+10h] [rbp-40h] BYREF
  __int64 v12; // [rsp+18h] [rbp-38h]

  v5 = (unsigned __int64 **)(a2 + 8);
  if ( a4 )
  {
    if ( !a3
      || (*(_BYTE *)(*(_QWORD *)(a1[23] + 200) + 904LL) & 0x40) == 0
      || (v10 = a4, v8 = sub_3220AA0(a1[26]), result = sub_E06A90(a3), a4 = v10, v8 >= (unsigned int)result) )
    {
      v12 = a4;
      HIWORD(v11) = 1;
      LODWORD(v11) = 4;
      WORD2(v11) = a3;
      return sub_3248F80(v5, a1 + 11, &v11);
    }
  }
  else if ( !a3
         || (*(_BYTE *)(*(_QWORD *)(a1[23] + 200) + 904LL) & 0x40) == 0
         || (v9 = (unsigned __int16)sub_3220AA0(a1[26]), result = sub_E06A90(a3), v9 >= (unsigned int)result) )
  {
    WORD2(v11) = a3;
    LODWORD(v11) = 1;
    HIWORD(v11) = 1;
    v12 = 0;
    return sub_3248F80(v5, a1 + 11, &v11);
  }
  return result;
}
