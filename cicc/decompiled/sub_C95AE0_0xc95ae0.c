// Function: sub_C95AE0
// Address: 0xc95ae0
//
__int64 __fastcall sub_C95AE0(__int64 a1)
{
  __int64 result; // rax
  unsigned int v2; // r8d
  int v3; // edx
  cpu_set_t cpuset; // [rsp+0h] [rbp-90h] BYREF

  if ( !*(_BYTE *)(a1 + 4) )
  {
    LODWORD(result) = sub_C95A80();
    v2 = *(_DWORD *)a1;
    if ( (int)result <= 0 )
      LODWORD(result) = 1;
    if ( v2 )
      goto LABEL_6;
    return (unsigned int)result;
  }
  if ( sched_getaffinity(0, 0x80u, &cpuset) )
  {
    v3 = sub_22420F0();
    LODWORD(result) = 1;
    if ( v3 > 0 )
      LODWORD(result) = v3;
  }
  else
  {
    LODWORD(result) = __sched_cpucount(0x80u, &cpuset);
    if ( (int)result <= 0 )
      LODWORD(result) = 1;
  }
  v2 = *(_DWORD *)a1;
  if ( !*(_DWORD *)a1 )
    return (unsigned int)result;
LABEL_6:
  if ( *(_BYTE *)(a1 + 5) )
  {
    if ( v2 > (unsigned int)result )
      return (unsigned int)result;
  }
  return v2;
}
