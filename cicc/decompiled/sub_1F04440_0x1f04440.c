// Function: sub_1F04440
// Address: 0x1f04440
//
__int64 __fastcall sub_1F04440(__int64 a1, _DWORD *a2)
{
  unsigned __int64 v2; // rax
  unsigned __int16 v4; // dx

  v2 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 24LL) + 16LL * (a2[2] & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
  if ( !*(_BYTE *)(v2 + 29) )
    return 0xFFFFFFFFLL;
  v4 = (*a2 >> 8) & 0xFFF;
  if ( v4 )
    return *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 248LL) + 4LL * v4);
  else
    return *(unsigned int *)(v2 + 24);
}
