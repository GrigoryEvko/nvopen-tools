// Function: sub_2977870
// Address: 0x2977870
//
__int64 __fastcall sub_2977870(__int64 a1, __int64 a2)
{
  _QWORD *v3; // r9
  _QWORD *v4; // rsi
  __int64 v5; // r8
  _QWORD *v6; // r9
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v10; // r8
  __int64 v11[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_BB9630(a2, a2);
  nullsub_79();
  sub_BB9660(a2, (__int64)&unk_4F86530);
  sub_BB9660(a2, (__int64)&unk_4F8144C);
  sub_BB9660(a2, (__int64)&unk_4F875EC);
  v3 = *(_QWORD **)(a2 + 112);
  v11[0] = (__int64)&unk_4F8144C;
  v4 = &v3[*(unsigned int *)(a2 + 120)];
  if ( v4 == sub_29777B0(v3, (__int64)v4, v11) )
  {
    if ( v5 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v5 + 1, 8u, v5, (__int64)v6);
      v4 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v4 = &unk_4F8144C;
    v6 = *(_QWORD **)(a2 + 112);
    v10 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v10;
    v4 = &v6[v10];
  }
  v11[0] = (__int64)&unk_4F875EC;
  if ( v4 == sub_29777B0(v6, (__int64)v4, v11) )
  {
    if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v7 + 1, 8u, v7, v8);
      v4 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v4 = &unk_4F875EC;
    ++*(_DWORD *)(a2 + 120);
  }
  return sub_BB9660(a2, (__int64)&unk_4F8D474);
}
