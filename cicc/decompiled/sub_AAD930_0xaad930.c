// Function: sub_AAD930
// Address: 0xaad930
//
bool __fastcall sub_AAD930(__int64 a1, __int64 a2)
{
  __int64 *v2; // r14
  unsigned int v3; // ebx
  __int64 v4; // rax
  __int64 v6; // r13

  v2 = *(__int64 **)a1;
  v3 = *(_DWORD *)(a1 + 8);
  if ( v3 <= 0x40 )
  {
    v4 = 0;
    if ( v3 )
      v4 = (__int64)((_QWORD)v2 << (64 - (unsigned __int8)v3)) >> (64 - (unsigned __int8)v3);
    return a2 < v4;
  }
  v6 = v2[(v3 - 1) >> 6] & (1LL << ((unsigned __int8)v3 - 1));
  if ( v6 )
  {
    if ( v3 + 1 - (unsigned int)sub_C44500(a1) > 0x40 )
      return v6 == 0;
LABEL_9:
    v4 = *v2;
    return a2 < v4;
  }
  if ( v3 + 1 - (unsigned int)sub_C444A0(a1) <= 0x40 )
    goto LABEL_9;
  return v6 == 0;
}
