// Function: sub_2F318B0
// Address: 0x2f318b0
//
void __fastcall sub_2F318B0(__int64 a1, __int64 a2)
{
  _QWORD *v4; // rdi
  _QWORD *v5; // rsi
  __int64 v6; // r8
  __int64 v7; // r9
  _QWORD *v8; // r9
  __int64 v9; // r8
  _QWORD *v10; // rsi
  __int64 v11; // r8
  _QWORD *v12; // r9
  __int64 v13; // r8
  _QWORD *v14; // r9
  __int64 v15; // r8
  _QWORD *v16; // r9
  __int64 v17; // r8
  _QWORD *v18; // r9
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // r8
  __int64 v22; // r8
  __int64 v23; // r8
  __int64 v24; // r8
  __int64 v25[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_QWORD **)(a2 + 144);
  v5 = &v4[*(unsigned int *)(a2 + 152)];
  v25[0] = (__int64)&unk_501EB14;
  if ( v5 == sub_2F310C0(v4, (__int64)v5, v25) )
  {
    if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 156) )
    {
      sub_C8D5F0(a2 + 144, (const void *)(a2 + 160), v6 + 1, 8u, v6, v7);
      v5 = (_QWORD *)(*(_QWORD *)(a2 + 144) + 8LL * *(unsigned int *)(a2 + 152));
    }
    *v5 = &unk_501EB14;
    ++*(_DWORD *)(a2 + 152);
  }
  v8 = *(_QWORD **)(a2 + 112);
  v9 = *(unsigned int *)(a2 + 120);
  v25[0] = (__int64)&unk_501EB14;
  v10 = &v8[v9];
  if ( v10 == sub_2F310C0(v8, (__int64)v10, v25) )
  {
    if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v11 + 1, 8u, v11, (__int64)v12);
      v10 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v10 = &unk_501EB14;
    v12 = *(_QWORD **)(a2 + 112);
    v24 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v24;
    v10 = &v12[v24];
  }
  v25[0] = (__int64)&unk_5025C1C;
  if ( v10 == sub_2F310C0(v12, (__int64)v10, v25) )
  {
    if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v13 + 1, 8u, v13, (__int64)v14);
      v10 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v10 = &unk_5025C1C;
    v14 = *(_QWORD **)(a2 + 112);
    v23 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v23;
    v10 = &v14[v23];
  }
  v25[0] = (__int64)&unk_501EACC;
  if ( v10 == sub_2F310C0(v14, (__int64)v10, v25) )
  {
    if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v15 + 1, 8u, v15, (__int64)v16);
      v10 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v10 = &unk_501EACC;
    v16 = *(_QWORD **)(a2 + 112);
    v22 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v22;
    v10 = &v16[v22];
  }
  v25[0] = (__int64)&unk_501FE44;
  if ( v10 == sub_2F310C0(v16, (__int64)v10, v25) )
  {
    if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v17 + 1, 8u, v17, (__int64)v18);
      v10 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v10 = &unk_501FE44;
    v18 = *(_QWORD **)(a2 + 112);
    v21 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v21;
    v10 = &v18[v21];
  }
  v25[0] = (__int64)&unk_50208AC;
  if ( v10 == sub_2F310C0(v18, (__int64)v10, v25) )
  {
    if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v19 + 1, 8u, v19, v20);
      v10 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v10 = &unk_50208AC;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_2E84680(a1, a2);
}
