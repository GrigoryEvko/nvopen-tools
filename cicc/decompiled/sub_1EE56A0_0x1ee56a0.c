// Function: sub_1EE56A0
// Address: 0x1ee56a0
//
__int64 __fastcall sub_1EE56A0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rax
  unsigned int v3; // r8d

  v2 = (__int64 *)sub_1DB3C70((__int64 *)a1, a2);
  v3 = 0;
  if ( v2 != (__int64 *)(*(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8)) )
    LOBYTE(v3) = (*(_DWORD *)((*v2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v2 >> 1) & 3) <= (*(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a2 >> 1) & 3);
  return v3;
}
