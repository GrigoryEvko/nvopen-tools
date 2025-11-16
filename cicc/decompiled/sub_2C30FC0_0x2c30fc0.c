// Function: sub_2C30FC0
// Address: 0x2c30fc0
//
__int64 __fastcall sub_2C30FC0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rsi
  __int64 result; // rax

  while ( 1 )
  {
    v6 = a1[13];
    v7 = a1[12];
    v8 = a1[28];
    result = a1[29] - v8;
    if ( v6 - v7 != result )
      goto LABEL_2;
    if ( v7 == v6 )
      return result;
    while ( *(_QWORD *)v7 == *(_QWORD *)v8 )
    {
      result = *(unsigned __int8 *)(v7 + 24);
      if ( (_BYTE)result != *(_BYTE *)(v8 + 24) )
        break;
      if ( (_BYTE)result )
      {
        if ( *(_QWORD *)(v7 + 8) != *(_QWORD *)(v8 + 8) )
          break;
        result = *(_QWORD *)(v8 + 16);
        if ( *(_QWORD *)(v7 + 16) != result )
          break;
      }
      v7 += 32;
      v8 += 32;
      if ( v6 == v7 )
        return result;
    }
LABEL_2:
    result = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v6 - 32) + 8LL) - 1;
    if ( (unsigned int)result <= 1 )
      return result;
    sub_2AD7320((__int64)a1, v8, v6, v7, a5, a6);
  }
}
