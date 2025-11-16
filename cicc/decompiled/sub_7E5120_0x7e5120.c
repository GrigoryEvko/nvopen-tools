// Function: sub_7E5120
// Address: 0x7e5120
//
__int64 __fastcall sub_7E5120(__int64 a1)
{
  int v1; // eax
  __int64 result; // rax
  _QWORD *i; // rcx
  __int64 v4; // rdx

  v1 = *(unsigned __int8 *)(a1 + 174);
  *(_BYTE *)(a1 + 205) |= 2u;
  result = (unsigned int)(v1 - 1);
  if ( (unsigned __int8)result <= 1u )
  {
    for ( i = *(_QWORD **)(a1 + 176); i; i = (_QWORD *)*i )
    {
      v4 = i[1];
      if ( !*(_BYTE *)(v4 + 172) )
        *(_BYTE *)(v4 + 205) |= 2u;
      for ( result = *(_QWORD *)(v4 + 112); result; result = *(_QWORD *)(result + 112) )
      {
        if ( v4 != *(_QWORD *)(result + 272) )
          break;
        if ( !*(_BYTE *)(result + 172) )
          *(_BYTE *)(result + 205) |= 2u;
      }
    }
  }
  return result;
}
