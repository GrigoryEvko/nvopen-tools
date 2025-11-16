// Function: sub_BC85F0
// Address: 0xbc85f0
//
__int64 __fastcall sub_BC85F0(__int64 a1, const char *a2)
{
  unsigned __int8 v2; // al
  _BYTE **v3; // rax
  _BYTE *v4; // r14
  unsigned int v5; // r15d
  size_t v6; // r13
  const void *v7; // rax
  __int64 v8; // rdx

  v2 = *(_BYTE *)(a1 - 16);
  if ( (v2 & 2) != 0 )
    v3 = *(_BYTE ***)(a1 - 32);
  else
    v3 = (_BYTE **)(a1 - 8LL * ((v2 >> 2) & 0xF) - 16);
  v4 = *v3;
  v5 = 0;
  if ( !**v3 )
  {
    v6 = strlen(a2);
    v7 = (const void *)sub_B91420((__int64)v4);
    if ( v6 == v8 )
    {
      v5 = 1;
      if ( v6 )
        LOBYTE(v5) = memcmp(v7, a2, v6) == 0;
    }
  }
  return v5;
}
