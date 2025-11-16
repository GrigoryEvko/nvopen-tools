// Function: sub_28FDFC0
// Address: 0x28fdfc0
//
__int64 __fastcall sub_28FDFC0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rdi
  _QWORD *v4; // rsi
  __int64 v5; // r8
  __int64 v6; // r9
  _QWORD *v7; // rdi
  _QWORD *v8; // rsi
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v12[5]; // [rsp+8h] [rbp-28h] BYREF

  *(_BYTE *)(a2 + 160) = 1;
  v3 = *(_QWORD **)(a2 + 112);
  v4 = &v3[*(unsigned int *)(a2 + 120)];
  v12[0] = (__int64)&unk_4F8144C;
  if ( v4 == sub_28FDF00(v3, (__int64)v4, v12) )
  {
    if ( v5 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v5 + 1, 8u, v5, v6);
      v4 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v4 = &unk_4F8144C;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB9660(a2, (__int64)&unk_4F8144C);
  v7 = *(_QWORD **)(a2 + 112);
  v8 = &v7[*(unsigned int *)(a2 + 120)];
  v12[0] = (__int64)&unk_4F875EC;
  if ( v8 == sub_28FDF00(v7, (__int64)v8, v12) )
  {
    if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v9 + 1, 8u, v9, v10);
      v8 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v8 = &unk_4F875EC;
    ++*(_DWORD *)(a2 + 120);
  }
  return sub_BB9660(a2, (__int64)&unk_4F875EC);
}
