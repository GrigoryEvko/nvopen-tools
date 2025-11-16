// Function: sub_7370D0
// Address: 0x7370d0
//
__int64 __fastcall sub_7370D0(__int64 a1, unsigned int a2)
{
  __int64 v2; // rax
  bool v3; // dl
  __int64 result; // rax
  char v5; // cl
  char v6; // dl
  char v7; // al
  char v8; // dl

  v2 = *(_QWORD *)(a1 + 40);
  v3 = 0;
  if ( v2 )
    v3 = *(_BYTE *)(v2 + 28) == 1;
  result = *(unsigned __int8 *)(a1 + 144);
  if ( (result & 8) == 0 )
  {
    result &= 4u;
    if ( a2 || !(_BYTE)result )
    {
      v5 = *(_BYTE *)(a1 + 140);
      if ( (unsigned __int8)(v5 - 9) <= 2u )
      {
        if ( v3 )
          goto LABEL_10;
        if ( !a2 )
          return result;
      }
      else if ( v5 == 2 )
      {
        if ( (*(_BYTE *)(a1 + 161) & 8) == 0 || v3 )
          goto LABEL_10;
      }
      else if ( v5 != 12 || (_BYTE)result != 0 || *(_QWORD *)(a1 + 8) == 0 || v3 )
      {
LABEL_10:
        v6 = *(_BYTE *)(a1 + 144);
        v7 = v6 | 0xC;
        v8 = v6 | 4;
        if ( !a2 )
          v7 = v8;
        *(_BYTE *)(a1 + 144) = v7;
        return sub_736EA0(a1, a2);
      }
      JUMPOUT(0x737010);
    }
  }
  return result;
}
