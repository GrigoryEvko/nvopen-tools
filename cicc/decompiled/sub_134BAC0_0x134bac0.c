// Function: sub_134BAC0
// Address: 0x134bac0
//
__int64 __fastcall sub_134BAC0(__int64 a1)
{
  _QWORD *v2; // r12
  _QWORD *v3; // rbx
  _QWORD *v4; // rdi

  v2 = (_QWORD *)(a1 + 1024);
  v3 = (_QWORD *)a1;
  do
  {
    v4 = v3;
    v3 += 2;
    sub_1348A90(v4);
  }
  while ( v2 != v3 );
  *(_QWORD *)(a1 + 1024) = 0;
  *(_QWORD *)(a1 + 1048) = 0;
  *(_OWORD *)(a1 + 1032) = 0;
  *(_QWORD *)(a1 + 1056) = 0;
  *(_QWORD *)(a1 + 4216) = 0;
  memset(
    (void *)((a1 + 1064) & 0xFFFFFFFFFFFFFFF8LL),
    0,
    8LL * (((unsigned int)a1 - (((_DWORD)a1 + 1064) & 0xFFFFFFF8) + 4224) >> 3));
  *(_QWORD *)(a1 + 4224) = 0;
  *(_QWORD *)(a1 + 4232) = 0;
  *(_QWORD *)(a1 + 5248) = 0;
  memset(
    (void *)((a1 + 4240) & 0xFFFFFFFFFFFFFFF8LL),
    0,
    8LL * (((_DWORD)a1 + 4232 - (((_DWORD)a1 + 4240) & 0xFFFFFFF8) + 1024) >> 3));
  *(_OWORD *)(a1 + 5256) = 0;
  *(_QWORD *)(a1 + 5272) = 0;
  return 0;
}
