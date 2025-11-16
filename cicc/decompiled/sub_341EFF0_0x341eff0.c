// Function: sub_341EFF0
// Address: 0x341eff0
//
void __fastcall sub_341EFF0(__int64 a1, __int64 a2)
{
  int v4; // ebx
  _QWORD *v5; // rdi
  _QWORD *v6; // rsi
  __int64 v7; // r8
  __int64 v8; // r9
  _QWORD *v9; // rdi
  _QWORD *v10; // rsi
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13[7]; // [rsp+8h] [rbp-38h] BYREF

  v4 = *(_DWORD *)(*(_QWORD *)(a1 + 200) + 792LL);
  if ( v4 )
    sub_BB9660(a2, (__int64)&unk_4F86530);
  sub_BB9660(a2, (__int64)&unk_501DA08);
  v5 = *(_QWORD **)(a2 + 112);
  v13[0] = (__int64)&unk_501DA08;
  v6 = &v5[*(unsigned int *)(a2 + 120)];
  if ( v6 == sub_341EA90(v5, (__int64)v6, v13) )
  {
    if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v7 + 1, 8u, v7, v8);
      v6 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v6 = &unk_501DA08;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB9660(a2, (__int64)&unk_4F6D3F0);
  sub_BB9660(a2, (__int64)&unk_4F8662C);
  if ( (_BYTE)qword_5039C48 && v4 )
    sub_BB9660(a2, (__int64)&unk_4F8E808);
  sub_BB9660(a2, (__int64)&unk_4F87C64);
  sub_BB9660(a2, (__int64)&unk_50165D0);
  v9 = *(_QWORD **)(a2 + 112);
  v13[0] = (__int64)&unk_50165D0;
  v10 = &v9[*(unsigned int *)(a2 + 120)];
  if ( v10 == sub_341EA90(v9, (__int64)v10, v13) )
  {
    if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v11 + 1, 8u, v11, v12);
      v10 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v10 = &unk_50165D0;
    ++*(_DWORD *)(a2 + 120);
  }
  if ( v4 )
    sub_1027A20(a2);
  sub_2E84680(a1, a2);
}
