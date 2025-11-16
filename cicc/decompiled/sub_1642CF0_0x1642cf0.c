// Function: sub_1642CF0
// Address: 0x1642cf0
//
bool __fastcall sub_1642CF0(__int64 a1)
{
  unsigned __int64 v1; // rax
  __int64 *v2; // rdx
  __int64 *v3; // rax
  __int64 v4; // rax

  v1 = a1 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (a1 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    return 0;
  v2 = (__int64 *)(v1 - 24);
  v3 = (__int64 *)(v1 - 72);
  if ( (a1 & 4) != 0 )
    v3 = v2;
  v4 = *v3;
  return !*(_BYTE *)(v4 + 16) && *(_DWORD *)(v4 + 36) == 78;
}
