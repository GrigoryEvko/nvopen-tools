// Function: sub_2E4F420
// Address: 0x2e4f420
//
__int64 __fastcall sub_2E4F420(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rdi
  _QWORD *v4; // rsi
  __int64 v5; // r8
  __int64 v6; // r9
  _QWORD *v7; // rdi
  _QWORD *v8; // rsi
  __int64 v9; // r8
  __int64 v10; // r9
  _QWORD *v11; // rdi
  _QWORD *v12; // rsi
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v16[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_BB9630(a2, a2);
  sub_2E84680(a1, a2);
  v3 = *(_QWORD **)(a2 + 112);
  v4 = &v3[*(unsigned int *)(a2 + 120)];
  v16[0] = (__int64)&unk_50208AC;
  if ( v4 == sub_2E4F080(v3, (__int64)v4, v16) )
  {
    if ( v5 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v5 + 1, 8u, v5, v6);
      v4 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v4 = &unk_50208AC;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB9660(a2, (__int64)&unk_501FE44);
  v7 = *(_QWORD **)(a2 + 112);
  v16[0] = (__int64)&unk_501FE44;
  v8 = &v7[*(unsigned int *)(a2 + 120)];
  if ( v8 == sub_2E4F080(v7, (__int64)v8, v16) )
  {
    if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v9 + 1, 8u, v9, v10);
      v8 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v8 = &unk_501FE44;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB9660(a2, (__int64)&unk_501EC08);
  v11 = *(_QWORD **)(a2 + 112);
  v16[0] = (__int64)&unk_501EC08;
  v12 = &v11[*(unsigned int *)(a2 + 120)];
  if ( v12 == sub_2E4F080(v11, (__int64)v12, v16) )
  {
    if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v13 + 1, 8u, v13, v14);
      v12 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v12 = &unk_501EC08;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB9660(a2, (__int64)&unk_502D274);
  return sub_BB9660(a2, (__int64)&unk_4F89C28);
}
