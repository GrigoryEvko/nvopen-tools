// Function: sub_8C6310
// Address: 0x8c6310
//
__int64 __fastcall sub_8C6310(__int64 a1)
{
  __int64 result; // rax
  char v2; // dl
  __int64 v3; // rdx

  result = a1;
  if ( a1 )
  {
    v2 = *(_BYTE *)(a1 + 140);
    if ( (unsigned __int8)(v2 - 9) > 2u )
      goto LABEL_9;
LABEL_3:
    v3 = *(_QWORD *)(result + 168);
    if ( (*(_BYTE *)(v3 + 109) & 0x28) != 0 || (*(_BYTE *)(result + 177) & 0x90) == 0x10 && *(_QWORD *)(v3 + 168) )
    {
      do
      {
        result = *(_QWORD *)(result + 112);
        if ( !result )
          break;
        v2 = *(_BYTE *)(result + 140);
        if ( (unsigned __int8)(v2 - 9) <= 2u )
          goto LABEL_3;
LABEL_9:
        ;
      }
      while ( v2 == 12 && (*(_DWORD *)(result + 184) & 0x4000FF) == 0xA );
    }
  }
  return result;
}
