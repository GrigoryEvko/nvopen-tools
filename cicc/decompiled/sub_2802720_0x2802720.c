// Function: sub_2802720
// Address: 0x2802720
//
__int64 __fastcall sub_2802720(__int64 a1, __int64 a2)
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
  _QWORD *v15; // rdi
  _QWORD *v16; // rsi
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v20[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_BB9660(a2, (__int64)&unk_4F8662C);
  sub_BB9660(a2, (__int64)&unk_4F8144C);
  v3 = *(_QWORD **)(a2 + 112);
  v20[0] = (__int64)&unk_4F8144C;
  v4 = &v3[*(unsigned int *)(a2 + 120)];
  if ( v4 == sub_2802660(v3, (__int64)v4, v20) )
  {
    if ( v5 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v5 + 1, 8u, v5, v6);
      v4 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v4 = &unk_4F8144C;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB9660(a2, (__int64)&unk_4F875EC);
  v7 = *(_QWORD **)(a2 + 112);
  v20[0] = (__int64)&unk_4F875EC;
  v8 = &v7[*(unsigned int *)(a2 + 120)];
  if ( v8 == sub_2802660(v7, (__int64)v8, v20) )
  {
    if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v9 + 1, 8u, v9, v10);
      v8 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v8 = &unk_4F875EC;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB9660(a2, (__int64)&unk_4F8C12C);
  v11 = *(_QWORD **)(a2 + 112);
  v20[0] = (__int64)&unk_4F8C12C;
  v12 = &v11[*(unsigned int *)(a2 + 120)];
  if ( v12 == sub_2802660(v11, (__int64)v12, v20) )
  {
    if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v13 + 1, 8u, v13, v14);
      v12 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v12 = &unk_4F8C12C;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB9660(a2, (__int64)&unk_4F8FAE4);
  sub_BB9660(a2, (__int64)&unk_4F881C8);
  v15 = *(_QWORD **)(a2 + 112);
  v20[0] = (__int64)&unk_4F881C8;
  v16 = &v15[*(unsigned int *)(a2 + 120)];
  if ( v16 == sub_2802660(v15, (__int64)v16, v20) )
  {
    if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v17 + 1, 8u, v17, v18);
      v16 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v16 = &unk_4F881C8;
    ++*(_DWORD *)(a2 + 120);
  }
  return sub_BB9660(a2, (__int64)&unk_4F89C28);
}
