// Function: sub_2FCF730
// Address: 0x2fcf730
//
void __fastcall sub_2FCF730(__int64 a1, __int64 a2)
{
  _QWORD *v4; // rdi
  _QWORD *v5; // rsi
  __int64 v6; // r8
  __int64 v7; // r9
  _QWORD *v8; // rdi
  _QWORD *v9; // rsi
  __int64 v10; // r8
  __int64 v11; // r9
  _QWORD *v12; // r9
  _QWORD *v13; // rsi
  __int64 v14; // r8
  _QWORD *v15; // r9
  __int64 v16; // r8
  _QWORD *v17; // r9
  __int64 v18; // r8
  _QWORD *v19; // r9
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // r8
  __int64 v23; // r8
  __int64 v24; // r8
  __int64 v25[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_BB9630(a2, a2);
  sub_BB9660(a2, (__int64)&unk_5025C1C);
  v4 = *(_QWORD **)(a2 + 112);
  v25[0] = (__int64)&unk_5025C1C;
  v5 = &v4[*(unsigned int *)(a2 + 120)];
  if ( v5 == sub_2FCEC40(v4, (__int64)v5, v25) )
  {
    if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v6 + 1, 8u, v6, v7);
      v5 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v5 = &unk_5025C1C;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB9660(a2, (__int64)&unk_501EB0C);
  v8 = *(_QWORD **)(a2 + 112);
  v25[0] = (__int64)&unk_501EB0C;
  v9 = &v8[*(unsigned int *)(a2 + 120)];
  if ( v9 == sub_2FCEC40(v8, (__int64)v9, v25) )
  {
    if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v10 + 1, 8u, v10, v11);
      v9 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v9 = &unk_501EB0C;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB9660(a2, (__int64)&unk_501EC08);
  v12 = *(_QWORD **)(a2 + 112);
  v25[0] = (__int64)&unk_501EC08;
  v13 = &v12[*(unsigned int *)(a2 + 120)];
  if ( v13 == sub_2FCEC40(v12, (__int64)v13, v25) )
  {
    if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v14 + 1, 8u, v14, (__int64)v15);
      v13 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v13 = &unk_501EC08;
    v15 = *(_QWORD **)(a2 + 112);
    v24 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v24;
    v13 = &v15[v24];
  }
  v25[0] = (__int64)&unk_501FE44;
  if ( v13 == sub_2FCEC40(v15, (__int64)v13, v25) )
  {
    if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v16 + 1, 8u, v16, (__int64)v17);
      v13 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v13 = &unk_501FE44;
    v17 = *(_QWORD **)(a2 + 112);
    v23 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v23;
    v13 = &v17[v23];
  }
  v25[0] = (__int64)&unk_501EACC;
  if ( v13 == sub_2FCEC40(v17, (__int64)v13, v25) )
  {
    if ( v18 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v18 + 1, 8u, v18, (__int64)v19);
      v13 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v13 = &unk_501EACC;
    v19 = *(_QWORD **)(a2 + 112);
    v22 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v22;
    v13 = &v19[v22];
  }
  v25[0] = (__int64)&unk_501E91C;
  if ( v13 == sub_2FCEC40(v19, (__int64)v13, v25) )
  {
    if ( v20 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v20 + 1, 8u, v20, v21);
      v13 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v13 = &unk_501E91C;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_2E84680(a1, a2);
}
