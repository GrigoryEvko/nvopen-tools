// Function: sub_2168170
// Address: 0x2168170
//
void *__fastcall sub_2168170(
        _QWORD *a1,
        int a2,
        __int64 a3,
        _BYTE *a4,
        __int64 a5,
        int a6,
        __int128 a7,
        int *a8,
        __int64 a9,
        int a10)
{
  int v11; // [rsp+0h] [rbp-20h] BYREF
  int v13; // [rsp+8h] [rbp-18h] BYREF

  if ( *(_BYTE *)(a9 + 4) )
    v13 = *(_DWORD *)a9;
  if ( *((_BYTE *)a8 + 4) )
    v11 = *a8;
  sub_2167C50((__int64)a1, a2, a3, a4, a5, a6, a7, (int)&v11, &v13, a10, 1);
  *a1 = &unk_4A02B80;
  return &unk_4A02B80;
}
