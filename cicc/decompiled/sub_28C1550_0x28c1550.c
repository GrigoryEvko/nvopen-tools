// Function: sub_28C1550
// Address: 0x28c1550
//
__int64 __fastcall sub_28C1550(__int64 a1, __int64 a2)
{
  _QWORD *v3; // r9
  _QWORD *v4; // rsi
  __int64 v5; // r8
  _QWORD *v6; // r9
  __int64 v7; // r8
  _QWORD *v8; // r9
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v12; // r8
  __int64 v13; // r8
  __int64 v14[7]; // [rsp+8h] [rbp-38h] BYREF

  v3 = *(_QWORD **)(a2 + 112);
  v4 = &v3[*(unsigned int *)(a2 + 120)];
  v14[0] = (__int64)&unk_4F8144C;
  if ( v4 == sub_28C1490(v3, (__int64)v4, v14) )
  {
    if ( v5 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v5 + 1, 8u, v5, (__int64)v6);
      v4 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v4 = &unk_4F8144C;
    v6 = *(_QWORD **)(a2 + 112);
    v12 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v12;
    v4 = &v6[v12];
  }
  v14[0] = (__int64)&unk_4F881C8;
  if ( v4 == sub_28C1490(v6, (__int64)v4, v14) )
  {
    if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v7 + 1, 8u, v7, (__int64)v8);
      v4 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v4 = &unk_4F881C8;
    v8 = *(_QWORD **)(a2 + 112);
    v13 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v13;
    v4 = &v8[v13];
  }
  v14[0] = (__int64)&unk_4F6D3F0;
  if ( v4 == sub_28C1490(v8, (__int64)v4, v14) )
  {
    if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v9 + 1, 8u, v9, v10);
      v4 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v4 = &unk_4F6D3F0;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB9660(a2, (__int64)&unk_4F8662C);
  sub_BB9660(a2, (__int64)&unk_4F8144C);
  sub_BB9660(a2, (__int64)&unk_4F881C8);
  sub_BB9660(a2, (__int64)&unk_4F6D3F0);
  sub_BB9660(a2, (__int64)&unk_4F89C28);
  return sub_BB9630(a2, (__int64)&unk_4F89C28);
}
