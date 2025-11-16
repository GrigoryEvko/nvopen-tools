// Function: sub_114A160
// Address: 0x114a160
//
void __fastcall sub_114A160(__int64 a1, __int64 a2, __int64 a3)
{
  bool v4; // zf
  __int64 v5; // rbx
  unsigned int **v6; // r13
  unsigned __int64 v7; // rax
  _QWORD *v8; // rax
  __int64 v9; // r9
  __int64 v10; // r15
  __int64 v11; // rsi
  unsigned int *v12; // rbx
  __int64 v13; // r13
  __int64 v14; // rdx
  __int16 v15; // ax
  unsigned int *v16; // rbx
  unsigned int *v17; // r13
  char v18; // [rsp+Ch] [rbp-104h]
  char v19; // [rsp+10h] [rbp-100h]
  char v21[32]; // [rsp+20h] [rbp-F0h] BYREF
  __int16 v22; // [rsp+40h] [rbp-D0h]
  unsigned int *v23; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v24; // [rsp+58h] [rbp-B8h]
  _BYTE v25[176]; // [rsp+60h] [rbp-B0h] BYREF

  v4 = *(_QWORD *)(a2 + 48) == 0;
  v23 = (unsigned int *)v25;
  v5 = *(_QWORD *)(a2 - 32);
  v24 = 0x800000000LL;
  if ( !v4 || (*(_BYTE *)(a2 + 7) & 0x20) != 0 )
    sub_B9AA80(a2, (__int64)&v23);
  v6 = *(unsigned int ***)(a1 + 32);
  _BitScanReverse64(&v7, 1LL << (*(_WORD *)(a2 + 2) >> 1));
  v18 = *(_WORD *)(a2 + 2) & 1;
  v22 = 257;
  v19 = 63 - (v7 ^ 0x3F);
  v8 = sub_BD2C40(80, unk_3F10A10);
  v10 = (__int64)v8;
  if ( v8 )
    sub_B4D3C0((__int64)v8, a3, v5, v18, v19, v9, 0, 0);
  v11 = v10;
  (*(void (__fastcall **)(unsigned int *, __int64, char *, unsigned int *, unsigned int *))(*(_QWORD *)v6[11] + 16LL))(
    v6[11],
    v10,
    v21,
    v6[7],
    v6[8]);
  v12 = *v6;
  v13 = (__int64)&(*v6)[4 * *((unsigned int *)v6 + 2)];
  while ( (unsigned int *)v13 != v12 )
  {
    v14 = *((_QWORD *)v12 + 1);
    v11 = *v12;
    v12 += 4;
    sub_B99FD0(v10, v11, v14);
  }
  v15 = *(_WORD *)(a2 + 2) & 0x380;
  *(_BYTE *)(v10 + 72) = *(_BYTE *)(a2 + 72);
  *(_WORD *)(v10 + 2) = v15 | *(_WORD *)(v10 + 2) & 0xFC7F;
  v16 = v23;
  v17 = &v23[4 * (unsigned int)v24];
  if ( v23 != v17 )
  {
    do
    {
      v11 = *v16;
      switch ( (int)v11 )
      {
        case 0:
        case 1:
        case 2:
        case 3:
        case 5:
        case 7:
        case 8:
        case 9:
        case 10:
        case 25:
        case 38:
          sub_B99FD0(v10, v11, *((_QWORD *)v16 + 1));
          break;
        default:
          break;
      }
      v16 += 4;
    }
    while ( v17 != v16 );
    v17 = v23;
  }
  if ( v17 != (unsigned int *)v25 )
    _libc_free(v17, v11);
}
