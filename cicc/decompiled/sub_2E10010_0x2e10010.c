// Function: sub_2E10010
// Address: 0x2e10010
//
__int64 __fastcall sub_2E10010(__int64 a1, __int64 a2)
{
  _QWORD *v4; // rsi
  _QWORD *v5; // rdi
  __int64 v6; // r8
  _QWORD *v7; // r9
  __int64 v8; // r8
  __int64 v9; // r9
  _QWORD *v10; // r9
  _QWORD *v11; // rsi
  __int64 v12; // r8
  _QWORD *v13; // r9
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v17; // r8
  __int64 v18; // r8
  __int64 v19[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_BB9630(a2, a2);
  v4 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
  v5 = *(_QWORD **)(a2 + 112);
  v19[0] = (__int64)&unk_501EB14;
  if ( v4 == sub_2E0FF50(v5, (__int64)v4, v19) )
  {
    if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v6 + 1, 8u, v6, (__int64)v7);
      v4 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v4 = &unk_501EB14;
    v7 = *(_QWORD **)(a2 + 112);
    v17 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v17;
    v4 = &v7[v17];
  }
  v19[0] = (__int64)&unk_50208AC;
  if ( v4 == sub_2E0FF50(v7, (__int64)v4, v19) )
  {
    if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v8 + 1, 8u, v8, v9);
      v4 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v4 = &unk_50208AC;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB96F0(a2, (__int64)&unk_501FE44);
  v10 = *(_QWORD **)(a2 + 112);
  v19[0] = (__int64)&unk_501FE44;
  v11 = &v10[*(unsigned int *)(a2 + 120)];
  if ( v11 == sub_2E0FF50(v10, (__int64)v11, v19) )
  {
    if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v12 + 1, 8u, v12, (__int64)v13);
      v11 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v11 = &unk_501FE44;
    v13 = *(_QWORD **)(a2 + 112);
    v18 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v18;
    v11 = &v13[v18];
  }
  v19[0] = (__int64)&unk_5025C1C;
  if ( v11 == sub_2E0FF50(v13, (__int64)v11, v19) )
  {
    if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v14 + 1, 8u, v14, v15);
      v11 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v11 = &unk_5025C1C;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB96F0(a2, (__int64)&unk_5025C1C);
  return sub_2E84680(a1, a2);
}
