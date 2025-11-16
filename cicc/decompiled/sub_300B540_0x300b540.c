// Function: sub_300B540
// Address: 0x300b540
//
void __fastcall sub_300B540(__int64 a1, __int64 a2)
{
  _QWORD *v4; // rdi
  _QWORD *v5; // rsi
  __int64 v6; // r8
  __int64 v7; // r9
  _QWORD *v8; // rdi
  _QWORD *v9; // rsi
  __int64 v10; // r8
  __int64 v11; // r9
  _QWORD *v12; // rdi
  _QWORD *v13; // rsi
  __int64 v14; // r8
  __int64 v15; // r9
  _QWORD *v16; // rdi
  __int64 v17; // r8
  _QWORD *v18; // rsi
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21[7]; // [rsp+8h] [rbp-38h] BYREF

  sub_BB9630(a2, a2);
  sub_BB9660(a2, (__int64)&unk_501EACC);
  v4 = *(_QWORD **)(a2 + 112);
  v21[0] = (__int64)&unk_501EACC;
  v5 = &v4[*(unsigned int *)(a2 + 120)];
  if ( v5 == sub_300ADE0(v4, (__int64)v5, v21) )
  {
    if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v6 + 1, 8u, v6, v7);
      v5 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v5 = &unk_501EACC;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB9660(a2, (__int64)&unk_5025C1C);
  v8 = *(_QWORD **)(a2 + 112);
  v21[0] = (__int64)&unk_5025C1C;
  v9 = &v8[*(unsigned int *)(a2 + 120)];
  if ( v9 == sub_300ADE0(v8, (__int64)v9, v21) )
  {
    if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v10 + 1, 8u, v10, v11);
      v9 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v9 = &unk_5025C1C;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB9660(a2, (__int64)&unk_501E91C);
  sub_BB9660(a2, (__int64)&unk_501EB0C);
  v12 = *(_QWORD **)(a2 + 112);
  v21[0] = (__int64)&unk_501EB0C;
  v13 = &v12[*(unsigned int *)(a2 + 120)];
  if ( v13 == sub_300ADE0(v12, (__int64)v13, v21) )
  {
    if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v14 + 1, 8u, v14, v15);
      v13 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v13 = &unk_501EB0C;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB9660(a2, (__int64)&unk_502A66C);
  sub_BB9660(a2, (__int64)&unk_501EAFC);
  if ( !*(_BYTE *)(a1 + 304) )
  {
    v16 = *(_QWORD **)(a2 + 112);
    v17 = *(unsigned int *)(a2 + 120);
    v21[0] = (__int64)&unk_501E91C;
    v18 = &v16[v17];
    if ( v18 == sub_300ADE0(v16, (__int64)v18, v21) )
    {
      if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
      {
        sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v19 + 1, 8u, v19, v20);
        v18 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
      }
      *v18 = &unk_501E91C;
      ++*(_DWORD *)(a2 + 120);
    }
  }
  sub_2E84680(a1, a2);
}
