// Function: sub_15A9300
// Address: 0x15a9300
//
__int64 __fastcall sub_15A9300(__int64 a1, __int8 *a2, size_t a3)
{
  _DWORD *v5; // rbx
  unsigned int v6; // esi
  unsigned int v7; // edx
  unsigned int v8; // ecx
  unsigned int v9; // r8d
  unsigned int v10; // r8d

  v5 = &unk_4294D40;
  sub_15A9210(a1);
  *(_BYTE *)a1 = 0;
  v6 = 105;
  v7 = 1;
  *(_QWORD *)(a1 + 400) = 0;
  v8 = 1;
  v9 = 1;
  *(_QWORD *)(a1 + 4) = 0;
  *(_QWORD *)(a1 + 12) = 0;
  *(_DWORD *)(a1 + 416) = 0;
  while ( 1 )
  {
    sub_15A82D0(a1, v6, v7, v8, v9);
    if ( v5 + 2 == (_DWORD *)"DIFlagBlockByrefStruct" )
      break;
    v10 = v5[2];
    v8 = *((unsigned __int16 *)v5 + 7);
    v7 = *((unsigned __int16 *)v5 + 6);
    v6 = *((unsigned __int8 *)v5 + 8);
    v5 += 2;
    v9 = v10 >> 8;
  }
  sub_15A85E0(a1, 0, 8u, 8u, 8, 8);
  return sub_15A8830((_QWORD *)a1, a2, a3);
}
