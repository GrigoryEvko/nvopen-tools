// Function: sub_1EE5220
// Address: 0x1ee5220
//
bool __fastcall sub_1EE5220(__int64 a1, __int64 a2)
{
  __int64 *v3; // rdx
  unsigned __int64 v4; // rsi
  unsigned int v5; // eax
  int v6; // ecx
  bool result; // al

  v3 = (__int64 *)sub_1DB3C70((__int64 *)a1, a2);
  result = v3 != (__int64 *)(*(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8))
        && (v4 = a2 & 0xFFFFFFFFFFFFFFF8LL,
            v5 = *(_DWORD *)((*v3 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*v3 >> 1) & 3,
            v6 = *(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24),
            v5 <= (v6 | (unsigned int)(a2 >> 1) & 3))
        && v5 < (v6 | 1u)
        && (v4 | 6) != v3[1];
  return result;
}
