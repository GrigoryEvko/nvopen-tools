// Function: sub_2EEEC10
// Address: 0x2eeec10
//
void __fastcall sub_2EEEC10(__int64 a1, __int64 a2)
{
  _QWORD *v3; // r9
  _QWORD *v4; // rsi
  __int64 v5; // r8
  _QWORD *v6; // r9
  __int64 v7; // r8
  _QWORD *v8; // r9
  __int64 v9; // r8
  _QWORD *v10; // r9
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // r8
  __int64 v14; // r8
  __int64 v15; // r8
  __int64 v16[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = *(_QWORD **)(a2 + 144);
  v4 = &v3[*(unsigned int *)(a2 + 152)];
  v16[0] = (__int64)&unk_501EB0C;
  if ( v4 == sub_2EEE570(v3, (__int64)v4, v16) )
  {
    if ( v5 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 156) )
    {
      sub_C8D5F0(a2 + 144, (const void *)(a2 + 160), v5 + 1, 8u, v5, (__int64)v6);
      v4 = (_QWORD *)(*(_QWORD *)(a2 + 144) + 8LL * *(unsigned int *)(a2 + 152));
    }
    *v4 = &unk_501EB0C;
    v6 = *(_QWORD **)(a2 + 144);
    v13 = (unsigned int)(*(_DWORD *)(a2 + 152) + 1);
    *(_DWORD *)(a2 + 152) = v13;
    v4 = &v6[v13];
  }
  v16[0] = (__int64)&unk_501EB14;
  if ( v4 == sub_2EEE570(v6, (__int64)v4, v16) )
  {
    if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 156) )
    {
      sub_C8D5F0(a2 + 144, (const void *)(a2 + 160), v7 + 1, 8u, v7, (__int64)v8);
      v4 = (_QWORD *)(*(_QWORD *)(a2 + 144) + 8LL * *(unsigned int *)(a2 + 152));
    }
    *v4 = &unk_501EB14;
    v8 = *(_QWORD **)(a2 + 144);
    v15 = (unsigned int)(*(_DWORD *)(a2 + 152) + 1);
    *(_DWORD *)(a2 + 152) = v15;
    v4 = &v8[v15];
  }
  v16[0] = (__int64)&unk_5025C1C;
  if ( v4 == sub_2EEE570(v8, (__int64)v4, v16) )
  {
    if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 156) )
    {
      sub_C8D5F0(a2 + 144, (const void *)(a2 + 160), v9 + 1, 8u, v9, (__int64)v10);
      v4 = (_QWORD *)(*(_QWORD *)(a2 + 144) + 8LL * *(unsigned int *)(a2 + 152));
    }
    *v4 = &unk_5025C1C;
    v10 = *(_QWORD **)(a2 + 144);
    v14 = (unsigned int)(*(_DWORD *)(a2 + 152) + 1);
    *(_DWORD *)(a2 + 152) = v14;
    v4 = &v10[v14];
  }
  v16[0] = (__int64)&unk_501EACC;
  if ( v4 == sub_2EEE570(v10, (__int64)v4, v16) )
  {
    if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 156) )
    {
      sub_C8D5F0(a2 + 144, (const void *)(a2 + 160), v11 + 1, 8u, v11, v12);
      v4 = (_QWORD *)(*(_QWORD *)(a2 + 144) + 8LL * *(unsigned int *)(a2 + 152));
    }
    *v4 = &unk_501EACC;
    ++*(_DWORD *)(a2 + 152);
  }
  *(_BYTE *)(a2 + 160) = 1;
  sub_2E84680(a1, a2);
}
