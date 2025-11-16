// Function: sub_B842C0
// Address: 0xb842c0
//
char *__fastcall sub_B842C0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  char **v3; // r14
  __int64 v4; // r13
  __int64 v6; // rax
  _QWORD *v7; // rdi
  unsigned __int64 v8; // rcx
  __int64 v9; // rax

  v2 = a1 + 128;
  v3 = (char **)(a1 + 8);
  v4 = a1 + 48;
  *(_QWORD *)a1 = &unk_49DA9C0;
  *(_QWORD *)(a1 + 40) = 0x800000000LL;
  *(_QWORD *)(a1 + 120) = 0x800000000LL;
  v6 = a1 + 272;
  v7 = (_QWORD *)(a1 + 416);
  *(v7 - 51) = 0;
  *(v7 - 50) = 0;
  *(v7 - 49) = 0;
  *(v7 - 48) = v4;
  *(v7 - 38) = v2;
  *(v7 - 28) = 0;
  *(v7 - 27) = 0;
  *(v7 - 26) = 0;
  *((_DWORD *)v7 - 50) = 0;
  *(v7 - 24) = 0;
  *(v7 - 23) = 0;
  *(v7 - 22) = 0;
  *((_DWORD *)v7 - 42) = 0;
  *(v7 - 2) = 0;
  *(v7 - 1) = 1;
  *(v7 - 20) = v6;
  *(v7 - 19) = 0x1000000000LL;
  do
  {
    if ( v7 )
      *v7 = -4096;
    v7 += 2;
  }
  while ( v7 != (_QWORD *)(a1 + 544) );
  sub_C656D0(v7, 6);
  v8 = *(unsigned int *)(a1 + 44);
  *(_QWORD *)(a1 + 560) = 0;
  *(_QWORD *)(a1 + 576) = a1 + 592;
  *(_QWORD *)(a1 + 584) = 0x400000000LL;
  *(_QWORD *)(a1 + 624) = a1 + 640;
  v9 = *(unsigned int *)(a1 + 40);
  *(_QWORD *)(a1 + 568) = 0;
  *(_QWORD *)(a1 + 632) = 0;
  *(_QWORD *)(a1 + 640) = 0;
  *(_QWORD *)(a1 + 648) = 0;
  *(_QWORD *)(a1 + 656) = 0;
  *(_QWORD *)(a1 + 688) = 0;
  *(_QWORD *)(a1 + 664) = 0;
  *(_QWORD *)(a1 + 672) = 0;
  *(_DWORD *)(a1 + 680) = 0;
  *(_QWORD *)(a1 + 696) = 0;
  *(_QWORD *)(a1 + 704) = 0;
  *(_DWORD *)(a1 + 712) = 0;
  *(_QWORD *)(a2 + 8) = a1;
  if ( v9 + 1 > v8 )
  {
    sub_C8D5F0(a1 + 32, v4, v9 + 1, 8);
    v9 = *(unsigned int *)(a1 + 40);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v9) = a2;
  ++*(_DWORD *)(a1 + 40);
  return sub_B841D0(v3, a2);
}
