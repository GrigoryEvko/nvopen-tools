// Function: sub_7A6510
// Address: 0x7a6510
//
__int64 __fastcall sub_7A6510(__int64 a1, unsigned int *a2)
{
  unsigned int v2; // edx
  __int64 i; // rax
  __int64 result; // rax

  v2 = *(_DWORD *)(a1 + 140);
  if ( v2 )
  {
    for ( i = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    if ( !HIDWORD(qword_4F077B4) || v2 >= *a2 || (*(_BYTE *)(a1 + 144) & 1) != 0 || (*(_BYTE *)(i + 179) & 0x20) != 0 )
    {
      *a2 = v2;
      return 1;
    }
    else
    {
      sub_684B30(0x488u, (_DWORD *)(a1 + 64));
      *(_DWORD *)(a1 + 140) = *a2;
      return 1;
    }
  }
  else
  {
    result = 0;
    if ( (*(_BYTE *)(a1 + 144) & 1) != 0 )
    {
      *a2 = 1;
      return 1;
    }
  }
  return result;
}
