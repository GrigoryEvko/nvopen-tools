// Function: sub_1F20000
// Address: 0x1f20000
//
__int64 __fastcall sub_1F20000(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // r14
  __int64 *v4; // rdx
  int *v5; // r14
  unsigned __int64 *v6; // rax
  __int64 v7; // r13
  __int64 v8; // r8
  int v9; // r9d

  v2 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 272LL) + 392LL) + 16LL * *(unsigned int *)(a2 + 48));
  v3 = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL);
  v4 = (__int64 *)sub_1DB3C70((__int64 *)v3, v2);
  if ( v4 == (__int64 *)(*(_QWORD *)v3 + 24LL * *(unsigned int *)(v3 + 8)) )
    return v2;
  if ( (*(_DWORD *)((*v4 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v4 >> 1) & 3) > (*(_DWORD *)((v2 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                        | (unsigned int)(v2 >> 1) & 3) )
    return v2;
  v5 = (int *)v4[2];
  if ( !v5 )
    return v2;
  v6 = (unsigned __int64 *)sub_1DD5E10(a2, *(_QWORD *)(a2 + 32));
  v7 = sub_1F1AD70((_QWORD *)a1, 0, v5, v2, a2, v6);
  sub_1F1FA40(a1 + 200, v2, *(_QWORD *)(v7 + 8), *(unsigned int *)(a1 + 80), v8, v9);
  return *(_QWORD *)(v7 + 8);
}
