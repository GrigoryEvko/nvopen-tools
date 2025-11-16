// Function: sub_BC96B0
// Address: 0xbc96b0
//
_BYTE *__fastcall sub_BC96B0(__int64 a1, const char *a2)
{
  unsigned __int8 v2; // al
  __int64 v3; // rdi
  _BYTE *v4; // r15
  _BYTE *v5; // r14
  size_t v6; // r13
  const void *v7; // rax
  __int64 v8; // rdx
  __int64 v10; // rdi

  v2 = *(_BYTE *)(a1 - 16);
  if ( (v2 & 2) != 0 )
  {
    v3 = *(_QWORD *)(a1 - 32);
    v4 = *(_BYTE **)v3;
    if ( **(_BYTE **)v3 )
      return 0;
    v5 = *(_BYTE **)(v3 + 8);
    if ( *v5 != 1 )
      return 0;
  }
  else
  {
    v10 = a1 - 16 - 8LL * ((v2 >> 2) & 0xF);
    v4 = *(_BYTE **)v10;
    if ( **(_BYTE **)v10 )
      return 0;
    v5 = *(_BYTE **)(v10 + 8);
    if ( *v5 != 1 )
      return 0;
  }
  v6 = strlen(a2);
  v7 = (const void *)sub_B91420((__int64)v4);
  if ( v6 != v8 )
    return 0;
  if ( !v6 )
    return v5;
  if ( memcmp(v7, a2, v6) )
    return 0;
  return v5;
}
