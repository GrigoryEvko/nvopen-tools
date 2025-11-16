// Function: sub_2FBA740
// Address: 0x2fba740
//
unsigned __int64 __fastcall sub_2FBA740(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r14
  __int64 v3; // r15
  __int64 *v4; // rdx
  int *v5; // r10
  __int64 v6; // r15
  char v7; // al
  __int64 v8; // rax
  int *v10; // [rsp+8h] [rbp-38h]

  v2 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v3 = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL);
  v4 = (__int64 *)sub_2E09D00((__int64 *)v3, a2 & 0xFFFFFFFFFFFFFFF8LL | 6);
  if ( v4 == (__int64 *)(*(_QWORD *)v3 + 24LL * *(unsigned int *)(v3 + 8)) )
    return *(_QWORD *)(v2 + 8) & 0xFFFFFFFFFFFFFFF9LL;
  if ( (*(_DWORD *)((*v4 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v4 >> 1) & 3) > (*(_DWORD *)(v2 + 24) | 3u) )
    return *(_QWORD *)(v2 + 8) & 0xFFFFFFFFFFFFFFF9LL;
  v5 = (int *)v4[2];
  if ( !v5 )
    return *(_QWORD *)(v2 + 8) & 0xFFFFFFFFFFFFFFF9LL;
  v6 = *(_QWORD *)(v2 + 16);
  if ( *(_DWORD *)(a1 + 84)
    && v2 != (*((_QWORD *)v5 + 1) & 0xFFFFFFFFFFFFFFF8LL)
    && (v10 = (int *)v4[2],
        v7 = sub_2E89D80(*(_QWORD *)(v2 + 16), *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL) + 112LL), 0),
        v5 = v10,
        v7) )
  {
    sub_2FB7E60(a1, 0, v10);
    sub_2FB9FE0((__int64 *)a1, 0, v10, a2, *(_QWORD *)(v6 + 24), (__int64 *)v6);
    return a2;
  }
  else
  {
    if ( !v6 )
      BUG();
    v8 = v6;
    if ( (*(_BYTE *)v6 & 4) == 0 && (*(_BYTE *)(v6 + 44) & 8) != 0 )
    {
      do
        v8 = *(_QWORD *)(v8 + 8);
      while ( (*(_BYTE *)(v8 + 44) & 8) != 0 );
    }
    return *(_QWORD *)(sub_2FB9FE0(
                         (__int64 *)a1,
                         0,
                         v5,
                         a2 & 0xFFFFFFFFFFFFFFF8LL | 6,
                         *(_QWORD *)(v6 + 24),
                         *(__int64 **)(v8 + 8))
                     + 8);
  }
}
