// Function: sub_2207590
// Address: 0x2207590
//
__int64 __fastcall sub_2207590(__int64 a1)
{
  __int64 v1; // rbp
  unsigned __int32 v2; // eax
  __int64 v4; // rcx
  signed __int32 v5; // eax
  unsigned __int32 v6; // ett

  if ( *(_BYTE *)a1 )
    return 0;
  if ( &_pthread_key_create )
  {
    v2 = _InterlockedCompareExchange((volatile signed __int32 *)a1, 256, 0);
    if ( !v2 )
      return 1;
    v4 = v2;
    while ( (_DWORD)v4 != 1 )
    {
      if ( (_DWORD)v4 != 256 )
        goto LABEL_14;
      v6 = v2;
      v5 = _InterlockedCompareExchange((volatile signed __int32 *)a1, 65792, v2);
      v4 = 65792;
      if ( v6 == v5 )
        goto LABEL_14;
      if ( v5 == 1 )
        return 0;
      if ( v5 )
LABEL_14:
        syscall(202, a1, 0, v4, 0);
      v2 = _InterlockedCompareExchange((volatile signed __int32 *)a1, 256, 0);
      v4 = v2;
      if ( !v2 )
        return 1;
    }
  }
  else if ( !*(_BYTE *)a1 )
  {
    if ( *(_BYTE *)(a1 + 1) )
    {
      v1 = sub_2252770(8);
      sub_22076D0(v1);
      sub_2253480(v1, &`typeinfo for'__gnu_cxx::recursive_init_error, sub_2207690);
    }
    *(_BYTE *)(a1 + 1) = 1;
    return 1;
  }
  return 0;
}
