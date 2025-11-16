// Function: sub_7F9D00
// Address: 0x7f9d00
//
_BOOL8 __fastcall sub_7F9D00(__int64 a1)
{
  __int64 v1; // rbx

  v1 = *(_QWORD *)(*(_QWORD *)(a1 + 56) + 320LL);
  if ( !v1 )
    v1 = *(_QWORD *)(a1 + 56);
  return (*(_BYTE *)(a1 + 72) & 4) != 0
      && !(unsigned int)sub_72F880((_BYTE *)v1)
      && (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v1 + 40) + 32LL) + 179LL) & 4) != 0;
}
