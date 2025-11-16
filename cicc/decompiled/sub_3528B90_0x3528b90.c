// Function: sub_3528B90
// Address: 0x3528b90
//
void __fastcall sub_3528B90(__int64 a1, __int64 a2)
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
  __int64 v16[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_BB9630(a2, a2);
  v4 = *(_QWORD **)(a2 + 112);
  v5 = &v4[*(unsigned int *)(a2 + 120)];
  v16[0] = (__int64)&unk_501FE44;
  if ( v5 == sub_3528790(v4, (__int64)v5, v16) )
  {
    if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v6 + 1, 8u, v6, v7);
      v5 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v5 = &unk_501FE44;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB9660(a2, (__int64)&unk_50208AC);
  v8 = *(_QWORD **)(a2 + 112);
  v16[0] = (__int64)&unk_50208AC;
  v9 = &v8[*(unsigned int *)(a2 + 120)];
  if ( v9 == sub_3528790(v8, (__int64)v9, v16) )
  {
    if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v10 + 1, 8u, v10, v11);
      v9 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v9 = &unk_50208AC;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB9660(a2, (__int64)&unk_502234C);
  v12 = *(_QWORD **)(a2 + 112);
  v16[0] = (__int64)&unk_502234C;
  v13 = &v12[*(unsigned int *)(a2 + 120)];
  if ( v13 == sub_3528790(v12, (__int64)v13, v16) )
  {
    if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v14 + 1, 8u, v14, v15);
      v13 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v13 = &unk_502234C;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB9660(a2, (__int64)&unk_503BDA8);
  sub_BB9660(a2, (__int64)&unk_4F87C64);
  sub_2E84680(a1, a2);
}
