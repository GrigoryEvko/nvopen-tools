// Function: sub_1DB3E90
// Address: 0x1db3e90
//
__int64 __fastcall sub_1DB3E90(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v5; // r13
  __int64 v6; // r15
  __int64 *v7; // r12
  __int64 v8; // r14
  __int64 v9; // rdx
  __int64 v10; // rcx
  unsigned int v11; // esi
  unsigned int v12; // eax
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 *v15; // rax

  if ( !*(_DWORD *)(a2 + 8) )
    return 0;
  v5 = (__int64 *)sub_1DB3C70((__int64 *)a1, **(_QWORD **)a2);
  v6 = *(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8);
  if ( v5 == (__int64 *)v6 )
    return 0;
  v7 = (__int64 *)sub_1DB3C70((__int64 *)a2, *v5);
  v8 = *(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8);
  if ( (__int64 *)v8 == v7 )
    return 0;
LABEL_5:
  v9 = *v7;
  v10 = (*v7 >> 1) & 3;
  v11 = v10 | *(_DWORD *)((*v7 & 0xFFFFFFFFFFFFFFF8LL) + 24);
  v12 = *(_DWORD *)((v5[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v5[1] >> 1) & 3;
  if ( v11 >= v12 )
  {
LABEL_13:
    if ( (*(_DWORD *)((v7[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v7[1] >> 1) & 3) > v12 )
    {
      v14 = v6;
      v6 = v8;
      v8 = v14;
      v15 = v5;
      v5 = v7;
      v7 = v15;
    }
    while ( 1 )
    {
      v7 += 3;
      if ( v7 == (__int64 *)v8 )
        return 0;
      if ( (*(_DWORD *)((v7[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v7[1] >> 1) & 3) >= (*(_DWORD *)((*v5 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                                 | (unsigned int)(*v5 >> 1)
                                                                                                 & 3) )
        goto LABEL_5;
    }
  }
  if ( v11 <= ((unsigned int)(*v5 >> 1) & 3 | *(_DWORD *)((*v5 & 0xFFFFFFFFFFFFFFF8LL) + 24)) )
  {
    v10 = (*v5 >> 1) & 3;
    v9 = *v5;
  }
  if ( v10 )
  {
    v13 = 0;
    if ( (v9 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      v13 = *(_QWORD *)((v9 & 0xFFFFFFFFFFFFFFF8LL) + 16);
    if ( (unsigned __int8)sub_1EDB0A0(a3, v13) )
    {
      v12 = *(_DWORD *)((v5[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v5[1] >> 1) & 3;
      goto LABEL_13;
    }
  }
  return 1;
}
