// Function: sub_6EA170
// Address: 0x6ea170
//
__int64 __fastcall sub_6EA170(__int64 a1, _QWORD *a2)
{
  __int64 v3; // rax
  __int64 v4; // rbx

  *a2 = 0;
  if ( *(_BYTE *)(a1 + 16) != 1 )
    return 0;
  v3 = *(_QWORD *)(a1 + 144);
  if ( (*(_BYTE *)(v3 + 25) & 1) == 0 || *(_BYTE *)(v3 + 24) != 1 || *(_BYTE *)(v3 + 56) != 4 )
    return 0;
  v4 = *(_QWORD *)(v3 + 72);
  if ( *(_BYTE *)(v4 + 24) != 3 || !(unsigned int)sub_8D3110(*(_QWORD *)(*(_QWORD *)(v4 + 56) + 120LL)) )
    return 0;
  *a2 = *(_QWORD *)(v4 + 56);
  return 1;
}
