// Function: sub_1F1BC20
// Address: 0x1f1bc20
//
unsigned __int64 __fastcall sub_1F1BC20(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r14
  __int64 v3; // r15
  __int64 *v4; // rdx
  int *v5; // r10
  int v6; // eax
  __int64 v7; // r15
  char v8; // al
  __int64 v9; // rax
  int *v11; // [rsp+8h] [rbp-38h]
  int *v12; // [rsp+8h] [rbp-38h]

  v2 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v3 = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL);
  v4 = (__int64 *)sub_1DB3C70((__int64 *)v3, a2 & 0xFFFFFFFFFFFFFFF8LL | 6);
  if ( v4 == (__int64 *)(*(_QWORD *)v3 + 24LL * *(unsigned int *)(v3 + 8)) )
    return *(_QWORD *)(v2 + 8) & 0xFFFFFFFFFFFFFFF9LL;
  if ( (*(_DWORD *)((*v4 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v4 >> 1) & 3) > (*(_DWORD *)(v2 + 24) | 3u) )
    return *(_QWORD *)(v2 + 8) & 0xFFFFFFFFFFFFFFF9LL;
  v5 = (int *)v4[2];
  if ( !v5 )
    return *(_QWORD *)(v2 + 8) & 0xFFFFFFFFFFFFFFF9LL;
  v6 = *(_DWORD *)(a1 + 84);
  if ( !v2 )
  {
    if ( v6 )
    {
      if ( (*((_QWORD *)v5 + 1) & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v12 = (int *)v4[2];
        if ( (unsigned __int8)sub_1E166B0(0, *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL) + 112LL), 0) )
        {
          sub_1F1B3E0(a1, 0, v12);
          BUG();
        }
      }
    }
    goto LABEL_20;
  }
  v7 = *(_QWORD *)(v2 + 16);
  if ( v6 )
  {
    if ( v2 != (*((_QWORD *)v5 + 1) & 0xFFFFFFFFFFFFFFF8LL) )
    {
      v11 = (int *)v4[2];
      v8 = sub_1E166B0(*(_QWORD *)(v2 + 16), *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL) + 112LL), 0);
      v5 = v11;
      if ( v8 )
      {
        sub_1F1B3E0(a1, 0, v11);
        sub_1F1AD70((_QWORD *)a1, 0, v11, a2, *(_QWORD *)(v7 + 24), (unsigned __int64 *)v7);
        return a2;
      }
    }
  }
  if ( !v7 )
LABEL_20:
    BUG();
  v9 = v7;
  if ( (*(_BYTE *)v7 & 4) == 0 && (*(_BYTE *)(v7 + 46) & 8) != 0 )
  {
    do
      v9 = *(_QWORD *)(v9 + 8);
    while ( (*(_BYTE *)(v9 + 46) & 8) != 0 );
  }
  return *(_QWORD *)(sub_1F1AD70(
                       (_QWORD *)a1,
                       0,
                       v5,
                       a2 & 0xFFFFFFFFFFFFFFF8LL | 6,
                       *(_QWORD *)(v7 + 24),
                       *(unsigned __int64 **)(v9 + 8))
                   + 8);
}
