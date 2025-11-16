// Function: sub_2F61810
// Address: 0x2f61810
//
void __fastcall sub_2F61810(__int64 a1, __int64 a2)
{
  _QWORD *v4; // rdi
  _QWORD *v5; // rsi
  __int64 v6; // r8
  __int64 v7; // r9
  _QWORD *v8; // r9
  _QWORD *v9; // rsi
  __int64 v10; // r8
  _QWORD *v11; // r9
  __int64 v12; // r8
  __int64 v13; // r9
  _QWORD *v14; // r9
  _QWORD *v15; // rsi
  __int64 v16; // r8
  _QWORD *v17; // r9
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // r8
  __int64 v21; // r8
  __int64 v22[7]; // [rsp+8h] [rbp-38h] BYREF

  sub_BB9630(a2, a2);
  v4 = *(_QWORD **)(a2 + 144);
  v5 = &v4[*(unsigned int *)(a2 + 152)];
  v22[0] = (__int64)&unk_5025C1C;
  if ( v5 == sub_2F60F90(v4, (__int64)v5, v22) )
  {
    if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 156) )
    {
      sub_C8D5F0(a2 + 144, (const void *)(a2 + 160), v6 + 1, 8u, v6, v7);
      v5 = (_QWORD *)(*(_QWORD *)(a2 + 144) + 8LL * *(unsigned int *)(a2 + 152));
    }
    *v5 = &unk_5025C1C;
    ++*(_DWORD *)(a2 + 152);
  }
  sub_BB9660(a2, (__int64)&unk_501EACC);
  v8 = *(_QWORD **)(a2 + 112);
  v22[0] = (__int64)&unk_501EACC;
  v9 = &v8[*(unsigned int *)(a2 + 120)];
  if ( v9 == sub_2F60F90(v8, (__int64)v9, v22) )
  {
    if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v10 + 1, 8u, v10, (__int64)v11);
      v9 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v9 = &unk_501EACC;
    v11 = *(_QWORD **)(a2 + 112);
    v21 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v21;
    v9 = &v11[v21];
  }
  v22[0] = (__int64)&unk_5025C1C;
  if ( v9 == sub_2F60F90(v11, (__int64)v9, v22) )
  {
    if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v12 + 1, 8u, v12, v13);
      v9 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v9 = &unk_5025C1C;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB9660(a2, (__int64)&unk_50208AC);
  v14 = *(_QWORD **)(a2 + 112);
  v22[0] = (__int64)&unk_50208AC;
  v15 = &v14[*(unsigned int *)(a2 + 120)];
  if ( v15 == sub_2F60F90(v14, (__int64)v15, v22) )
  {
    if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v16 + 1, 8u, v16, (__int64)v17);
      v15 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v15 = &unk_50208AC;
    v17 = *(_QWORD **)(a2 + 112);
    v20 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v20;
    v15 = &v17[v20];
  }
  v22[0] = (__int64)&unk_501FE44;
  if ( v15 == sub_2F60F90(v17, (__int64)v15, v22) )
  {
    if ( v18 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v18 + 1, 8u, v18, v19);
      v15 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v15 = &unk_501FE44;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_2E84680(a1, a2);
}
