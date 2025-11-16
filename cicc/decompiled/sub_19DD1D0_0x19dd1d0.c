// Function: sub_19DD1D0
// Address: 0x19dd1d0
//
__int64 __fastcall sub_19DD1D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax

  v6 = a2 + 112;
  v7 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v7 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(v6, (const void *)(a2 + 128), 0, 8, a5, a6);
    v7 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v7) = &unk_4F9E06C;
  v8 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
  *(_DWORD *)(a2 + 120) = v8;
  if ( *(_DWORD *)(a2 + 124) <= (unsigned int)v8 )
  {
    sub_16CD150(v6, (const void *)(a2 + 128), 0, 8, a5, a6);
    v8 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v8) = &unk_4F9A488;
  v9 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
  *(_DWORD *)(a2 + 120) = v9;
  if ( *(_DWORD *)(a2 + 124) <= (unsigned int)v9 )
  {
    sub_16CD150(v6, (const void *)(a2 + 128), 0, 8, a5, a6);
    v9 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v9) = &unk_4F9B6E8;
  ++*(_DWORD *)(a2 + 120);
  sub_1636A40(a2, (__int64)&unk_4F9D764);
  sub_1636A40(a2, (__int64)&unk_4F9E06C);
  sub_1636A40(a2, (__int64)&unk_4F9A488);
  sub_1636A40(a2, (__int64)&unk_4F9B6E8);
  sub_1636A40(a2, (__int64)&unk_4F9D3C0);
  return sub_1636A10(a2, (__int64)&unk_4F9D3C0);
}
