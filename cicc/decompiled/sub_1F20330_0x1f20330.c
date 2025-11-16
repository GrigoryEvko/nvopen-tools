// Function: sub_1F20330
// Address: 0x1f20330
//
void __fastcall sub_1F20330(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r14
  __int64 *v5; // rdx
  __int64 v6; // r8
  int v7; // r9d
  int *v8; // rdx

  v4 = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL);
  v5 = (__int64 *)sub_1DB3C70((__int64 *)v4, a2);
  if ( v5 != (__int64 *)(*(_QWORD *)v4 + 24LL * *(unsigned int *)(v4 + 8))
    && (*(_DWORD *)((*v5 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v5 >> 1) & 3) <= (*(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                         | (unsigned int)(a2 >> 1) & 3) )
  {
    v8 = (int *)v5[2];
    if ( v8 )
      sub_1F1B3E0(a1, 0, v8);
  }
  sub_1F1FA40(a1 + 200, a2, a3, *(unsigned int *)(a1 + 80), v6, v7);
}
