// Function: sub_EA97E0
// Address: 0xea97e0
//
bool __fastcall sub_EA97E0(__int64 a1, const void *a2, size_t a3)
{
  bool v3; // zf
  __int64 v4; // rbx
  __int64 v6; // r13
  _QWORD v8[6]; // [rsp+0h] [rbp-30h] BYREF

  v3 = *(_QWORD *)(a1 + 856) == 0;
  v8[0] = a2;
  v8[1] = a3;
  if ( !v3 )
    return a1 + 824 != sub_EA96F0(a1 + 816, (__int64)v8);
  v4 = *(_QWORD *)(a1 + 768);
  v6 = v4 + 16LL * *(unsigned int *)(a1 + 776);
  if ( v4 == v6 )
    return 0;
  while ( *(_QWORD *)(v4 + 8) != a3 || a3 && memcmp(*(const void **)v4, a2, a3) )
  {
    v4 += 16;
    if ( v6 == v4 )
      return 0;
  }
  return v6 != v4;
}
