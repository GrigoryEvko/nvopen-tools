// Function: sub_1833A50
// Address: 0x1833a50
//
__int64 __fastcall sub_1833A50(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r14
  __int64 v4; // rax
  __int64 v5; // rax
  _QWORD *v6; // rax
  unsigned __int64 v7; // rdi
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  _QWORD v11[9]; // [rsp+0h] [rbp-450h] BYREF
  char v12; // [rsp+48h] [rbp-408h]
  __int64 v13; // [rsp+50h] [rbp-400h]
  __int64 v14; // [rsp+318h] [rbp-138h]
  unsigned __int64 v15; // [rsp+320h] [rbp-130h]
  __int64 v16; // [rsp+380h] [rbp-D0h]
  unsigned __int64 v17; // [rsp+388h] [rbp-C8h]

  v2 = (_QWORD *)(a1 + 8);
  sub_1361980((__int64)v11, *(_QWORD *)a1, a2);
  if ( *(_BYTE *)(a1 + 1064) )
  {
    v9 = *(_QWORD *)(a1 + 912);
    if ( v9 != *(_QWORD *)(a1 + 904) )
      _libc_free(v9);
    v10 = *(_QWORD *)(a1 + 808);
    if ( v10 != *(_QWORD *)(a1 + 800) )
      _libc_free(v10);
    if ( (*(_BYTE *)(a1 + 80) & 1) == 0 )
      j___libc_free_0(*(_QWORD *)(a1 + 88));
  }
  v4 = v11[1];
  *(_BYTE *)(a1 + 1064) = 1;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 16) = v4;
  v5 = v11[2];
  *(_QWORD *)(a1 + 80) = 1;
  *(_QWORD *)(a1 + 24) = v5;
  *(_QWORD *)(a1 + 32) = v11[3];
  *(_QWORD *)(a1 + 40) = v11[4];
  *(_QWORD *)(a1 + 48) = v11[5];
  *(_QWORD *)(a1 + 56) = v11[6];
  *(_QWORD *)(a1 + 64) = v11[7];
  v6 = (_QWORD *)(a1 + 88);
  do
  {
    if ( v6 )
    {
      *v6 = -8;
      v6[1] = 0;
      v6[2] = 0;
      v6[3] = 0;
      v6[4] = 0;
      v6[5] = -8;
      v6[6] = 0;
      v6[7] = 0;
      v6[8] = 0;
      v6[9] = 0;
    }
    v6 += 11;
  }
  while ( v6 != (_QWORD *)(a1 + 792) );
  *(_QWORD *)(a1 + 792) = 0;
  v7 = v17;
  *(_QWORD *)(a1 + 800) = a1 + 832;
  *(_QWORD *)(a1 + 808) = a1 + 832;
  *(_QWORD *)(a1 + 816) = 8;
  *(_DWORD *)(a1 + 824) = 0;
  *(_QWORD *)(a1 + 896) = 0;
  *(_QWORD *)(a1 + 904) = a1 + 936;
  *(_QWORD *)(a1 + 912) = a1 + 936;
  *(_QWORD *)(a1 + 920) = 16;
  *(_DWORD *)(a1 + 928) = 0;
  if ( v7 != v16 )
    _libc_free(v7);
  if ( v15 != v14 )
    _libc_free(v15);
  if ( (v12 & 1) == 0 )
    j___libc_free_0(v13);
  sub_134DAF0((__int64)v11, *(_QWORD *)a1, a2, v2);
  if ( *(_BYTE *)(a1 + 1168) )
    sub_134CA00((_QWORD *)(a1 + 1072));
  *(_BYTE *)(a1 + 1168) = 1;
  sub_134C930(a1 + 1072, v11);
  sub_134CA00(v11);
  return a1 + 1072;
}
