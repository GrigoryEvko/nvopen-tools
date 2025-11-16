// Function: sub_1611190
// Address: 0x1611190
//
char *__fastcall sub_1611190(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  char **v3; // r14
  __int64 v4; // r13
  _QWORD *v6; // rax
  _QWORD *v7; // rdi
  __int64 v8; // rax

  v2 = a1 + 128;
  v3 = (char **)(a1 + 8);
  v4 = a1 + 48;
  *(_QWORD *)a1 = &unk_49EDA38;
  *(_QWORD *)(a1 + 40) = 0x800000000LL;
  *(_QWORD *)(a1 + 120) = 0x800000000LL;
  *(_QWORD *)(a1 + 256) = a1 + 272;
  *(_QWORD *)(a1 + 264) = 0x1000000000LL;
  v6 = (_QWORD *)(a1 + 416);
  v7 = (_QWORD *)(a1 + 544);
  *(v7 - 67) = 0;
  *(v7 - 66) = 0;
  *(v7 - 65) = 0;
  *(v7 - 64) = v4;
  *(v7 - 54) = v2;
  *(v7 - 44) = 0;
  *(v7 - 43) = 0;
  *(v7 - 42) = 0;
  *((_DWORD *)v7 - 82) = 0;
  *(v7 - 40) = 0;
  *(v7 - 39) = 0;
  *(v7 - 38) = 0;
  *((_DWORD *)v7 - 74) = 0;
  *(v7 - 18) = 0;
  *(v7 - 17) = 1;
  do
  {
    if ( v6 )
      *v6 = -4;
    v6 += 2;
  }
  while ( v7 != v6 );
  sub_16BD940(v7, 6);
  *(_QWORD *)(a1 + 568) = 0;
  *(_QWORD *)(a1 + 576) = 0;
  *(_QWORD *)(a1 + 640) = 0;
  *(_QWORD *)(a1 + 544) = &unk_49ED530;
  *(_QWORD *)(a1 + 584) = a1 + 600;
  *(_QWORD *)(a1 + 592) = 0x400000000LL;
  *(_QWORD *)(a1 + 632) = a1 + 648;
  v8 = *(unsigned int *)(a1 + 40);
  *(_QWORD *)(a1 + 648) = 0;
  *(_QWORD *)(a1 + 656) = 0;
  *(_QWORD *)(a1 + 672) = 0;
  *(_QWORD *)(a1 + 704) = 0;
  *(_QWORD *)(a1 + 680) = 0;
  *(_QWORD *)(a1 + 688) = 0;
  *(_DWORD *)(a1 + 696) = 0;
  *(_QWORD *)(a1 + 712) = 0;
  *(_QWORD *)(a1 + 720) = 0;
  *(_DWORD *)(a1 + 728) = 0;
  *(_QWORD *)(a2 + 16) = a1;
  if ( (unsigned int)v8 >= *(_DWORD *)(a1 + 44) )
  {
    sub_16CD150(a1 + 32, v4, 0, 8);
    v8 = *(unsigned int *)(a1 + 40);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v8) = a2;
  ++*(_DWORD *)(a1 + 40);
  return sub_16110B0(v3, a2);
}
