// Function: sub_1F1B1B0
// Address: 0x1f1b1b0
//
unsigned __int64 __fastcall sub_1F1B1B0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rsi
  __int64 v3; // rbx
  __int64 *v4; // rdx
  int *v5; // rdx

  v2 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v3 = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL);
  v4 = (__int64 *)sub_1DB3C70((__int64 *)v3, v2);
  if ( v4 == (__int64 *)(*(_QWORD *)v3 + 24LL * *(unsigned int *)(v3 + 8)) )
    return v2;
  if ( (*(_DWORD *)((*v4 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v4 >> 1) & 3) > *(_DWORD *)(v2 + 24) )
    return v2;
  v5 = (int *)v4[2];
  if ( !v5 )
    return v2;
  if ( !v2 )
    BUG();
  return *(_QWORD *)(sub_1F1AD70(
                       (_QWORD *)a1,
                       *(_DWORD *)(a1 + 80),
                       v5,
                       v2,
                       *(_QWORD *)(*(_QWORD *)(v2 + 16) + 24LL),
                       *(unsigned __int64 **)(v2 + 16))
                   + 8);
}
