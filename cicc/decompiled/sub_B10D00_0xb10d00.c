// Function: sub_B10D00
// Address: 0xb10d00
//
__int64 __fastcall sub_B10D00(__int64 a1)
{
  __int64 v1; // rax
  unsigned __int8 v2; // dl

  v1 = sub_B10CD0(a1);
  v2 = *(_BYTE *)(v1 - 16);
  if ( (v2 & 2) != 0 )
    return **(_QWORD **)(v1 - 32);
  else
    return *(_QWORD *)(v1 - 16 - 8LL * ((v2 >> 2) & 0xF));
}
