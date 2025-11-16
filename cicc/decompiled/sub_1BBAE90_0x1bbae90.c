// Function: sub_1BBAE90
// Address: 0x1bbae90
//
__int64 __fastcall sub_1BBAE90(__int64 a1, __int64 a2)
{
  __int64 v3; // r15
  void *v4; // rsi
  int v5; // r8d
  int v6; // r9d
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax

  v3 = a2 + 112;
  nullsub_571();
  sub_1636A40(a2, (__int64)&unk_4F9D764);
  sub_1636A40(a2, (__int64)&unk_4F9A488);
  sub_1636A40(a2, (__int64)&unk_4F96DB4);
  sub_1636A40(a2, (__int64)&unk_4F9D3C0);
  sub_1636A40(a2, (__int64)&unk_4F9920C);
  sub_1636A40(a2, (__int64)&unk_4F9E06C);
  sub_1636A40(a2, (__int64)&unk_4F98D2C);
  v4 = &unk_4F99CB0;
  sub_1636A40(a2, (__int64)&unk_4F99CB0);
  v7 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v7 >= *(_DWORD *)(a2 + 124) )
  {
    v4 = (void *)(a2 + 128);
    sub_16CD150(v3, (const void *)(a2 + 128), 0, 8, v5, v6);
    v7 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v7) = &unk_4F9920C;
  v8 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
  *(_DWORD *)(a2 + 120) = v8;
  if ( *(_DWORD *)(a2 + 124) <= (unsigned int)v8 )
  {
    v4 = (void *)(a2 + 128);
    sub_16CD150(v3, (const void *)(a2 + 128), 0, 8, v5, v6);
    v8 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v8) = &unk_4F9E06C;
  v9 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
  *(_DWORD *)(a2 + 120) = v9;
  if ( *(_DWORD *)(a2 + 124) <= (unsigned int)v9 )
  {
    v4 = (void *)(a2 + 128);
    sub_16CD150(v3, (const void *)(a2 + 128), 0, 8, v5, v6);
    v9 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v9) = &unk_4F96DB4;
  v10 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
  *(_DWORD *)(a2 + 120) = v10;
  if ( *(_DWORD *)(a2 + 124) <= (unsigned int)v10 )
  {
    v4 = (void *)(a2 + 128);
    sub_16CD150(v3, (const void *)(a2 + 128), 0, 8, v5, v6);
    v10 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v10) = &unk_4F98E5C;
  ++*(_DWORD *)(a2 + 120);
  return sub_1636A10(a2, (__int64)v4);
}
