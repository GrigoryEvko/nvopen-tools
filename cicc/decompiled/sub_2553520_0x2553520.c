// Function: sub_2553520
// Address: 0x2553520
//
char __fastcall sub_2553520(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 + 88);
  if ( v1 )
  {
    LOBYTE(v1) = v1 == 1 && *(_QWORD *)(*(_QWORD *)(a1 + 72) + 32LL) == 0x7FFFFFFF;
  }
  else if ( *(_DWORD *)(a1 + 8) == 1 )
  {
    LOBYTE(v1) = **(_QWORD **)a1 == 0x7FFFFFFF;
  }
  return v1;
}
