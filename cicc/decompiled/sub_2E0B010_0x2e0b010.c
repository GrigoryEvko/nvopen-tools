// Function: sub_2E0B010
// Address: 0x2e0b010
//
__int64 __fastcall sub_2E0B010(__int64 a1)
{
  __int64 *v1; // rcx
  unsigned int v2; // r8d
  __int64 i; // rdi
  __int64 v4; // rax
  __int64 v5; // rdx

  v1 = *(__int64 **)a1;
  v2 = 0;
  for ( i = *(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8);
        (__int64 *)i != v1;
        v2 += (*(_DWORD *)((v4 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v4 >> 1) & 3)
            - (*(_DWORD *)((v5 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v5 >> 1) & 3) )
  {
    v4 = v1[1];
    v5 = *v1;
    v1 += 3;
  }
  return v2;
}
