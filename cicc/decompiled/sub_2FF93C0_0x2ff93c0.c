// Function: sub_2FF93C0
// Address: 0x2ff93c0
//
void __fastcall sub_2FF93C0(__int64 a1, __int64 a2)
{
  _QWORD *v4; // rsi
  _QWORD *v5; // rdi
  __int64 v6; // r8
  _QWORD *v7; // r9
  __int64 v8; // r8
  __int64 v9; // r9
  _QWORD *v10; // r9
  __int64 v11; // r8
  _QWORD *v12; // rsi
  __int64 v13; // r8
  _QWORD *v14; // r9
  __int64 v15; // r8
  _QWORD *v16; // r9
  __int64 v17; // r8
  _QWORD *v18; // r9
  __int64 v19; // r8
  _QWORD *v20; // r9
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // r8
  __int64 v24; // r8
  __int64 v25; // r8
  __int64 v26; // r8
  __int64 v27; // r8
  __int64 v28[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_BB9630(a2, a2);
  v4 = (_QWORD *)(*(_QWORD *)(a2 + 144) + 8LL * *(unsigned int *)(a2 + 152));
  v5 = *(_QWORD **)(a2 + 144);
  v28[0] = (__int64)&unk_4F86530;
  if ( v4 == sub_2FF8B50(v5, (__int64)v4, v28) )
  {
    if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 156) )
    {
      sub_C8D5F0(a2 + 144, (const void *)(a2 + 160), v6 + 1, 8u, v6, (__int64)v7);
      v4 = (_QWORD *)(*(_QWORD *)(a2 + 144) + 8LL * *(unsigned int *)(a2 + 152));
    }
    *v4 = &unk_4F86530;
    v7 = *(_QWORD **)(a2 + 144);
    v23 = (unsigned int)(*(_DWORD *)(a2 + 152) + 1);
    *(_DWORD *)(a2 + 152) = v23;
    v4 = &v7[v23];
  }
  v28[0] = (__int64)&unk_501EB14;
  if ( v4 == sub_2FF8B50(v7, (__int64)v4, v28) )
  {
    if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 156) )
    {
      sub_C8D5F0(a2 + 144, (const void *)(a2 + 160), v8 + 1, 8u, v8, v9);
      v4 = (_QWORD *)(*(_QWORD *)(a2 + 144) + 8LL * *(unsigned int *)(a2 + 152));
    }
    *v4 = &unk_501EB14;
    ++*(_DWORD *)(a2 + 152);
  }
  v10 = *(_QWORD **)(a2 + 112);
  v11 = *(unsigned int *)(a2 + 120);
  v28[0] = (__int64)&unk_501EB14;
  v12 = &v10[v11];
  if ( v12 == sub_2FF8B50(v10, (__int64)v12, v28) )
  {
    if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v13 + 1, 8u, v13, (__int64)v14);
      v12 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v12 = &unk_501EB14;
    v14 = *(_QWORD **)(a2 + 112);
    v27 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v27;
    v12 = &v14[v27];
  }
  v28[0] = (__int64)&unk_5025C1C;
  if ( v12 == sub_2FF8B50(v14, (__int64)v12, v28) )
  {
    if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v15 + 1, 8u, v15, (__int64)v16);
      v12 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v12 = &unk_5025C1C;
    v16 = *(_QWORD **)(a2 + 112);
    v26 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v26;
    v12 = &v16[v26];
  }
  v28[0] = (__int64)&unk_501EACC;
  if ( v12 == sub_2FF8B50(v16, (__int64)v12, v28) )
  {
    if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v17 + 1, 8u, v17, (__int64)v18);
      v12 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v12 = &unk_501EACC;
    v18 = *(_QWORD **)(a2 + 112);
    v25 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v25;
    v12 = &v18[v25];
  }
  v28[0] = (__int64)&unk_50208AC;
  if ( v12 == sub_2FF8B50(v18, (__int64)v12, v28) )
  {
    if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v19 + 1, 8u, v19, (__int64)v20);
      v12 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v12 = &unk_50208AC;
    v20 = *(_QWORD **)(a2 + 112);
    v24 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v24;
    v12 = &v20[v24];
  }
  v28[0] = (__int64)&unk_501FE44;
  if ( v12 == sub_2FF8B50(v20, (__int64)v12, v28) )
  {
    if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v21 + 1, 8u, v21, v22);
      v12 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v12 = &unk_501FE44;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_2E84680(a1, a2);
}
