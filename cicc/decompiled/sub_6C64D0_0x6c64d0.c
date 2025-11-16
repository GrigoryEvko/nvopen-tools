// Function: sub_6C64D0
// Address: 0x6c64d0
//
__int64 __fastcall sub_6C64D0(__int64 a1, __int64 a2, __int64 a3, int a4, unsigned int a5, __int64 a6, __int64 a7)
{
  __int64 v9; // rbx
  int v10; // edx
  int v11; // r14d
  __int16 v12; // r13
  __int64 result; // rax
  __int64 v16; // [rsp+18h] [rbp-E8h]
  __int64 v17; // [rsp+28h] [rbp-D8h] BYREF
  _BYTE v18[208]; // [rsp+30h] [rbp-D0h] BYREF

  v9 = a1;
  v16 = *(_QWORD *)(a7 + 16);
  sub_6E2250(v18, &v17, 4, 1, v16, a7);
  if ( *(_BYTE *)(a1 + 140) == 12 )
  {
    do
      v9 = *(_QWORD *)(v9 + 160);
    while ( *(_BYTE *)(v9 + 140) == 12 );
  }
  v10 = 1024;
  if ( v16 && *(_QWORD *)v16 )
    v10 = (((*(_BYTE *)(*(_QWORD *)v16 + 80LL) - 7) & 0xFD) == 0) + 1024;
  sub_6C5750(
    *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v9 + 96LL) + 8LL),
    a3,
    a2,
    0,
    a4,
    ((*(_BYTE *)(a7 + 43) >> 4) ^ 1) & 1,
    0,
    v10,
    0,
    a5,
    a6,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    (__int64 *)(a7 + 8),
    0,
    0);
  v11 = dword_4F061D8;
  v12 = unk_4F061DC;
  sub_68A7B0(a7);
  result = sub_6E2C70(v17, 1, *(_QWORD *)(a7 + 16), a7);
  dword_4F061D8 = v11;
  unk_4F061DC = v12;
  return result;
}
