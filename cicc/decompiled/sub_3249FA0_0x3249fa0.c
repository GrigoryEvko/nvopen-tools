// Function: sub_3249FA0
// Address: 0x3249fa0
//
__int64 __fastcall sub_3249FA0(__int64 *a1, __int64 a2, __int16 a3)
{
  __int64 result; // rax
  unsigned int v6; // r15d
  unsigned int v7; // r15d
  __int64 v8[8]; // [rsp+0h] [rbp-40h] BYREF

  if ( (unsigned __int16)sub_3220AA0(a1[26]) > 3u )
  {
    if ( a3 )
    {
      if ( (*(_BYTE *)(*(_QWORD *)(a1[23] + 200) + 904LL) & 0x40) != 0 )
      {
        v6 = (unsigned __int16)sub_3220AA0(a1[26]);
        result = sub_E06A90(a3);
        if ( v6 < (unsigned int)result )
          return result;
      }
    }
    LODWORD(v8[0]) = 1;
    WORD2(v8[0]) = a3;
    HIWORD(v8[0]) = 25;
LABEL_5:
    v8[1] = 1;
    return sub_3248F80((unsigned __int64 **)(a2 + 8), a1 + 11, v8);
  }
  if ( !a3
    || (*(_BYTE *)(*(_QWORD *)(a1[23] + 200) + 904LL) & 0x40) == 0
    || (v7 = (unsigned __int16)sub_3220AA0(a1[26]), result = sub_E06A90(a3), v7 >= (unsigned int)result) )
  {
    LODWORD(v8[0]) = 1;
    WORD2(v8[0]) = a3;
    HIWORD(v8[0]) = 12;
    goto LABEL_5;
  }
  return result;
}
