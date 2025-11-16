// Function: sub_2EC1ED0
// Address: 0x2ec1ed0
//
void __fastcall sub_2EC1ED0(__int64 a1, __int64 a2)
{
  _QWORD *v4; // rdi
  _QWORD *v5; // rsi
  __int64 v6; // r8
  __int64 v7; // r9
  _QWORD *v8; // rdi
  _QWORD *v9; // rsi
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_BB9630(a2, a2);
  sub_BB9660(a2, (__int64)&unk_501FE44);
  sub_BB9660(a2, (__int64)&unk_50208AC);
  sub_BB9660(a2, (__int64)&unk_4F86530);
  sub_BB9660(a2, (__int64)&unk_5027190);
  sub_BB9660(a2, (__int64)&unk_5025C1C);
  v4 = *(_QWORD **)(a2 + 112);
  v12[0] = (__int64)&unk_5025C1C;
  v5 = &v4[*(unsigned int *)(a2 + 120)];
  if ( v5 == sub_2EC1350(v4, (__int64)v5, v12) )
  {
    if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v6 + 1, 8u, v6, v7);
      v5 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v5 = &unk_5025C1C;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB9660(a2, (__int64)&unk_501EACC);
  v8 = *(_QWORD **)(a2 + 112);
  v12[0] = (__int64)&unk_501EACC;
  v9 = &v8[*(unsigned int *)(a2 + 120)];
  if ( v9 == sub_2EC1350(v8, (__int64)v9, v12) )
  {
    if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v10 + 1, 8u, v10, v11);
      v9 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v9 = &unk_501EACC;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_2E84680(a1, a2);
}
