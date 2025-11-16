// Function: sub_809820
// Address: 0x809820
//
__int64 __fastcall sub_809820(__int64 a1)
{
  __int64 v1; // rdi
  __int64 v2; // rdi

  v1 = sub_8D2250(a1);
  if ( (unsigned __int8)(*(_BYTE *)(v1 + 140) - 9) <= 2u
    && (*(_BYTE *)(v1 + 177) & 0x10) != 0
    && (v2 = sub_880FA0(v1)) != 0 )
  {
    return *(_QWORD *)(*(_QWORD *)(sub_892920(v2) + 88) + 104LL);
  }
  else
  {
    return 0;
  }
}
