// Function: sub_BC97F0
// Address: 0xbc97f0
//
bool __fastcall sub_BC97F0(__int64 a1, const char *a2)
{
  unsigned __int8 v2; // al
  __int64 v3; // rdi
  _BYTE *v4; // r8
  _BYTE *v5; // r13
  __int64 v6; // rax
  __int64 v7; // rdx
  bool result; // al
  size_t v9; // r14
  const void *v10; // rdi
  __int64 v11; // rdx

  v2 = *(_BYTE *)(a1 - 16);
  if ( (v2 & 2) == 0 )
  {
    v3 = a1 - 16 - 8LL * ((v2 >> 2) & 0xF);
    v4 = *(_BYTE **)v3;
    if ( !**(_BYTE **)v3 )
      goto LABEL_3;
    return 0;
  }
  v3 = *(_QWORD *)(a1 - 32);
  v4 = *(_BYTE **)v3;
  if ( **(_BYTE **)v3 )
    return 0;
LABEL_3:
  v5 = *(_BYTE **)(v3 + 8);
  if ( *v5 )
    return 0;
  v6 = sub_B91420((__int64)v4);
  if ( v7 != 13
    || *(_QWORD *)v6 != 0x46656C69666F7250LL
    || *(_DWORD *)(v6 + 8) != 1634562671
    || *(_BYTE *)(v6 + 12) != 116 )
  {
    return 0;
  }
  v9 = strlen(a2);
  v10 = (const void *)sub_B91420((__int64)v5);
  result = 0;
  if ( v9 == v11 )
  {
    result = 1;
    if ( v9 )
      return memcmp(v10, a2, v9) == 0;
  }
  return result;
}
