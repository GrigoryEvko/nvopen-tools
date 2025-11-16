// Function: sub_2D174D0
// Address: 0x2d174d0
//
__int64 __fastcall sub_2D174D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r8
  char v7; // r13
  unsigned __int64 v8; // rbx
  unsigned __int64 v9; // rdi
  void *v10; // rsi
  unsigned __int64 v12[44]; // [rsp+0h] [rbp-160h] BYREF

  v6 = sub_BC1CD0(a4, &unk_4F81450, a3);
  memset(v12, 0, 0x128u);
  v12[7] = (unsigned __int64)&v12[5];
  v12[8] = (unsigned __int64)&v12[5];
  v12[13] = (unsigned __int64)&v12[11];
  v12[14] = (unsigned __int64)&v12[11];
  v12[19] = (unsigned __int64)&v12[17];
  v12[20] = (unsigned __int64)&v12[17];
  v12[25] = (unsigned __int64)&v12[23];
  v12[26] = (unsigned __int64)&v12[23];
  v12[31] = (unsigned __int64)&v12[29];
  v12[32] = (unsigned __int64)&v12[29];
  v7 = sub_2D13E90((__int64)v12, a3, v6 + 8);
  if ( v12[34] )
    j_j___libc_free_0(v12[34]);
  sub_2D0FFC0(v12[30]);
  sub_2D0FDF0(v12[24]);
  sub_2D0FDF0(v12[18]);
  sub_2D0FDF0(v12[12]);
  v8 = v12[6];
  while ( v8 )
  {
    sub_2D0F560(*(_QWORD *)(v8 + 24));
    v9 = v8;
    v8 = *(_QWORD *)(v8 + 16);
    j_j___libc_free_0(v9);
  }
  if ( v12[0] )
    j_j___libc_free_0(v12[0]);
  v10 = (void *)(a1 + 32);
  if ( v7 )
  {
    v12[1] = (unsigned __int64)&v12[4];
    v12[2] = 0x100000002LL;
    LODWORD(v12[3]) = 0;
    BYTE4(v12[3]) = 1;
    v12[6] = 0;
    v12[7] = (unsigned __int64)&v12[10];
    v12[8] = 2;
    LODWORD(v12[9]) = 0;
    BYTE4(v12[9]) = 1;
    v12[4] = (unsigned __int64)&unk_4F82408;
    v12[0] = 1;
    if ( &unk_4F82408 != (_UNKNOWN *)&qword_4F82400 && &unk_4F82408 != &unk_4F81450 )
    {
      HIDWORD(v12[2]) = 2;
      v12[5] = (unsigned __int64)&unk_4F81450;
      v12[0] = 2;
    }
    sub_C8CF70(a1, v10, 2, (__int64)&v12[4], (__int64)v12);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)&v12[10], (__int64)&v12[6]);
    if ( !BYTE4(v12[9]) )
      _libc_free(v12[7]);
    if ( !BYTE4(v12[3]) )
      _libc_free(v12[1]);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = v10;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
  }
  return a1;
}
