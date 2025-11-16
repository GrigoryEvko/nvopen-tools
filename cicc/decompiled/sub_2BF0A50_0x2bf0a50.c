// Function: sub_2BF0A50
// Address: 0x2bf0a50
//
unsigned __int64 __fastcall sub_2BF0A50(__int64 a1)
{
  unsigned __int64 v1; // rax

  v1 = *(_QWORD *)(a1 + 112) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v1 == a1 + 112 )
    return 0;
  if ( *(_DWORD *)(a1 + 88) != 2 )
  {
    if ( (!sub_2BF0A20(a1) || *(_BYTE *)(*(_QWORD *)(a1 + 48) + 128LL)) && *(_DWORD *)(a1 + 88) <= 2u )
      return 0;
    v1 = *(_QWORD *)(a1 + 112) & 0xFFFFFFFFFFFFFFF8LL;
  }
  if ( !v1 )
    return 0;
  return v1 - 24;
}
