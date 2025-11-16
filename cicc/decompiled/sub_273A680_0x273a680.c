// Function: sub_273A680
// Address: 0x273a680
//
__int64 __fastcall sub_273A680(__int64 a1)
{
  unsigned int v1; // eax
  __int64 *v2; // rbx
  unsigned int v3; // r8d
  __int64 v4; // rbx
  unsigned int v6; // r13d
  __int64 v7; // r12
  __int64 v8; // rdi
  int v9; // eax

  v1 = *(_DWORD *)(a1 + 32);
  v2 = *(__int64 **)(a1 + 24);
  if ( v1 > 0x40 )
  {
    v6 = v1 + 1;
    v7 = a1 + 24;
    v8 = a1 + 24;
    if ( (v2[(v1 - 1) >> 6] & (1LL << ((unsigned __int8)v1 - 1))) != 0 )
    {
      if ( v6 - (unsigned int)sub_C44500(v8) <= 0x40 )
      {
        v4 = *v2;
        if ( v4 != 0x8000000000000000LL )
        {
          v9 = sub_C44500(v7);
          v3 = 1;
          if ( v6 - v9 > 0x40 )
            return v3;
          goto LABEL_4;
        }
      }
    }
    else if ( v6 - (unsigned int)sub_C444A0(v8) <= 0x40 && *v2 != 0x8000000000000000LL )
    {
      v3 = sub_C444A0(v7);
      if ( v6 - v3 <= 0x40 )
      {
        v4 = *v2;
        goto LABEL_4;
      }
    }
  }
  else
  {
    v3 = 1;
    if ( !v1 )
      return v3;
    v4 = (__int64)((_QWORD)v2 << (64 - (unsigned __int8)v1)) >> (64 - (unsigned __int8)v1);
    if ( v4 != 0x8000000000000000LL )
    {
LABEL_4:
      LOBYTE(v3) = v4 != 0x7FFFFFFFFFFFFFFFLL;
      return v3;
    }
  }
  return 0;
}
