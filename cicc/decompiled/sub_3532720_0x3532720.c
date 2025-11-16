// Function: sub_3532720
// Address: 0x3532720
//
void __fastcall sub_3532720(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rdi
  _QWORD *v4; // rsi
  __int64 v5; // r8
  __int64 v6; // r9
  _QWORD *v7; // rdi
  _QWORD *v8; // rsi
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_BB9660(a2, (__int64)&unk_50208C0);
  v3 = *(_QWORD **)(a2 + 112);
  v11[0] = (__int64)&unk_50208C0;
  v4 = &v3[*(unsigned int *)(a2 + 120)];
  if ( v4 == sub_3532310(v3, (__int64)v4, v11) )
  {
    if ( v5 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v5 + 1, 8u, v5, v6);
      v4 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v4 = &unk_50208C0;
    ++*(_DWORD *)(a2 + 120);
  }
  v7 = *(_QWORD **)(a2 + 144);
  v8 = &v7[*(unsigned int *)(a2 + 152)];
  v11[0] = (__int64)&unk_4F8780C;
  if ( v8 == sub_3532310(v7, (__int64)v8, v11) )
  {
    if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 156) )
    {
      sub_C8D5F0(a2 + 144, (const void *)(a2 + 160), v9 + 1, 8u, v9, v10);
      v8 = (_QWORD *)(*(_QWORD *)(a2 + 144) + 8LL * *(unsigned int *)(a2 + 152));
    }
    *v8 = &unk_4F8780C;
    ++*(_DWORD *)(a2 + 152);
  }
  *(_BYTE *)(a2 + 160) = 1;
  nullsub_79();
}
