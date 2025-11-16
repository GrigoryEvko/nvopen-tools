// Function: sub_2DB2D10
// Address: 0x2db2d10
//
__int64 __fastcall sub_2DB2D10(__int64 a1, __int64 a2)
{
  _QWORD *v4; // rdi
  _QWORD *v5; // rsi
  __int64 v6; // r8
  __int64 v7; // r9
  _QWORD *v8; // rdi
  _QWORD *v9; // rsi
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v13[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_BB9660(a2, (__int64)&unk_501F1C8);
  sub_BB9660(a2, (__int64)&unk_501FE44);
  v4 = *(_QWORD **)(a2 + 112);
  v13[0] = (__int64)&unk_501FE44;
  v5 = &v4[*(unsigned int *)(a2 + 120)];
  if ( v5 == sub_2DB1FA0(v4, (__int64)v5, v13) )
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
  v13[0] = (__int64)&unk_50208AC;
  v9 = &v8[*(unsigned int *)(a2 + 120)];
  if ( v9 == sub_2DB1FA0(v8, (__int64)v9, v13) )
  {
    if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v10 + 1, 8u, v10, v11);
      v9 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v9 = &unk_50208AC;
    ++*(_DWORD *)(a2 + 120);
  }
  return sub_2E84680(a1, a2);
}
