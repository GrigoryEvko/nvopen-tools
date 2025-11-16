// Function: sub_1212CC0
// Address: 0x1212cc0
//
__int64 __fastcall sub_1212CC0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  _BYTE *v4; // rsi
  __int64 v5[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( (unsigned __int8)sub_120AFE0(a1, 460, "expected 'args' here")
    || (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here") )
  {
    return 1;
  }
  while ( 1 )
  {
    result = sub_120C050(a1, v5);
    if ( (_BYTE)result )
      break;
    v4 = *(_BYTE **)(a2 + 8);
    if ( v4 == *(_BYTE **)(a2 + 16) )
    {
      sub_9CA200(a2, v4, v5);
    }
    else
    {
      if ( v4 )
      {
        *(_QWORD *)v4 = v5[0];
        v4 = *(_BYTE **)(a2 + 8);
      }
      *(_QWORD *)(a2 + 8) = v4 + 8;
    }
    if ( *(_DWORD *)(a1 + 240) != 4 )
      return sub_120AFE0(a1, 13, "expected ')' here");
    *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  }
  return result;
}
