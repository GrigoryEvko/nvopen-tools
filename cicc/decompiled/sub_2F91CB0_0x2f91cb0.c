// Function: sub_2F91CB0
// Address: 0x2f91cb0
//
__int64 __fastcall sub_2F91CB0(__int64 a1, _DWORD *a2)
{
  unsigned __int64 v2; // rdx
  unsigned __int16 v4; // ax

  v2 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 56LL) + 16LL * (a2[2] & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
  if ( !*(_BYTE *)(v2 + 43) )
    return -1;
  v4 = (*a2 >> 8) & 0xFFF;
  if ( v4 )
    return *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 272LL) + 16LL * v4);
  else
    return *(_QWORD *)(v2 + 24);
}
