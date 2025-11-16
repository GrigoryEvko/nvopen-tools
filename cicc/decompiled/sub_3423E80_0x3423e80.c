// Function: sub_3423E80
// Address: 0x3423e80
//
char __fastcall sub_3423E80(__int64 a1, __int64 a2)
{
  int v2; // esi
  __int64 v3; // rax

  v2 = *(_DWORD *)(a2 + 24);
  if ( v2 < 0 )
  {
    return (*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 800) + 8LL) - 40LL * (unsigned int)~v2 + 24) >> 21) & 1LL;
  }
  else if ( v2 <= 499 )
  {
    if ( v2 > 239 )
    {
      LOBYTE(v3) = (unsigned int)(v2 - 242) <= 1;
    }
    else
    {
      LOBYTE(v3) = 1;
      if ( v2 <= 237 )
        LOBYTE(v3) = (unsigned int)(v2 - 101) <= 0x2F;
    }
  }
  else
  {
    LOBYTE(v3) = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 64) + 8LL) + 32LL))(*(_QWORD *)(*(_QWORD *)(a1 + 64) + 8LL));
  }
  return v3;
}
