// Function: sub_2F62EE0
// Address: 0x2f62ee0
//
unsigned __int64 __fastcall sub_2F62EE0(__int64 **a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 *v3; // rax
  __int64 v4; // r8
  unsigned __int64 v5; // r9
  unsigned __int64 result; // rax

  v2 = *a1[1];
  v3 = (__int64 *)sub_2E09D00((__int64 *)a2, v2);
  if ( v3 == (__int64 *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8)) )
    return sub_2E0E0B0(a2, *a1[1], *a1, 3LL * *(unsigned int *)(a2 + 8), v4, v5);
  result = *(_DWORD *)((*v3 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v3 >> 1) & 3;
  if ( (unsigned int)result > (*(_DWORD *)((v2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v2 >> 1) & 3) )
    return sub_2E0E0B0(a2, *a1[1], *a1, 3LL * *(unsigned int *)(a2 + 8), v4, v5);
  return result;
}
