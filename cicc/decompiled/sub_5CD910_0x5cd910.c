// Function: sub_5CD910
// Address: 0x5cd910
//
__int64 __fastcall sub_5CD910(_BYTE *a1, __int64 a2, char a3)
{
  unsigned __int8 v4; // di
  _BYTE *v6; // rax
  __int64 v7; // [rsp+8h] [rbp-18h] BYREF

  if ( a3 == 11 )
  {
    if ( a1[8] )
    {
      *(_BYTE *)(a2 + 195) |= 0x10u;
      if ( a1[9] == 3 && unk_4F077C4 == 2 )
      {
        sub_73EA10(a2 + 152, &v7);
        if ( !*(_QWORD *)(*(_QWORD *)(v7 + 168) + 56LL) )
        {
          v6 = (_BYTE *)sub_725E60();
          *v6 |= 0x19u;
          *(_QWORD *)(*(_QWORD *)(v7 + 168) + 56LL) = v6;
        }
      }
    }
  }
  else
  {
    v4 = 5;
    if ( a1[9] != 2 )
      v4 = (a1[11] & 0x10) == 0 ? 8 : 5;
    sub_5CCAE0(v4, (__int64)a1);
  }
  return a2;
}
