// Function: sub_1F14200
// Address: 0x1f14200
//
__int64 __fastcall sub_1F14200(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v7; // rdx
  __int64 *v8; // rcx
  __int64 v9; // rax
  __int64 v10; // rbx
  unsigned __int64 v11; // rax

  v7 = 16LL * *(unsigned int *)(a3 + 48);
  v8 = (__int64 *)(v7 + a1[1]);
  v9 = *v8;
  if ( (*v8 & 0xFFFFFFFFFFFFFFF8LL) == 0 || (v8[1] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    v9 = sub_1F13A50(a1, a2, a3, (__int64)v8, a5, a6);
    v7 = 16LL * *(unsigned int *)(a3 + 48);
  }
  if ( *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*a1 + 272LL) + 392LL) + v7 + 8) == v9 )
    return a3 + 24;
  v10 = 0;
  v11 = v9 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v11 )
    return *(_QWORD *)(v11 + 16);
  return v10;
}
