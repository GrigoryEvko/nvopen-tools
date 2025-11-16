// Function: sub_1DBCAA0
// Address: 0x1dbcaa0
//
__int64 __fastcall sub_1DBCAA0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r15
  __int64 v5; // rsi
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 v9; // rdx
  __int64 v10; // r13
  __int64 v11; // rbx
  __int64 *v12; // rsi
  __int64 v13; // rdi
  __int64 v14; // rax
  unsigned __int64 v15; // rbx
  int v16; // eax
  __int64 v18; // [rsp+8h] [rbp-48h]

  v4 = *(_QWORD *)(a2 + 64);
  v18 = v4 + 8LL * *(unsigned int *)(a2 + 72);
  if ( v4 == v18 )
    return 0;
  while ( 1 )
  {
    v5 = *(_QWORD *)(*(_QWORD *)v4 + 8LL);
    if ( (v5 & 0xFFFFFFFFFFFFFFF8LL) != 0 && (v5 & 6) == 0 )
    {
      v6 = *(_QWORD *)(a1 + 272);
      v7 = sub_1DA9310(v6, v5);
      v8 = *(_QWORD *)(v7 + 72);
      v9 = *(_QWORD *)(v7 + 64);
      if ( (unsigned int)((v8 - v9) >> 3) > 0x64 )
        return 1;
      if ( v8 != v9 )
        break;
    }
LABEL_3:
    v4 += 8;
    if ( v18 == v4 )
      return 0;
  }
  v10 = *(_QWORD *)(v7 + 64);
  while ( 1 )
  {
    v14 = *(_QWORD *)(*(_QWORD *)(v6 + 392) + 16LL * *(unsigned int *)(*(_QWORD *)v10 + 48LL) + 8);
    v15 = v14 & 0xFFFFFFFFFFFFFFF8LL;
    v16 = (v14 >> 1) & 3;
    v11 = v16 ? (2LL * (v16 - 1)) | v15 : *(_QWORD *)v15 & 0xFFFFFFFFFFFFFFF8LL | 6;
    v12 = (__int64 *)sub_1DB3C70((__int64 *)a2, v11);
    v13 = 0;
    if ( v12 != (__int64 *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8))
      && (*(_DWORD *)((*v12 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v12 >> 1) & 3) <= (*(_DWORD *)((v11 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                             | (unsigned int)(v11 >> 1)
                                                                                             & 3) )
    {
      v13 = v12[2];
    }
    if ( a3 == v13 )
      return 1;
    v10 += 8;
    if ( v8 == v10 )
      goto LABEL_3;
    v6 = *(_QWORD *)(a1 + 272);
  }
}
