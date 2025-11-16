// Function: sub_2ED5090
// Address: 0x2ed5090
//
__int64 __fastcall sub_2ED5090(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  unsigned int v10; // ecx
  __int16 *v11; // rsi
  int v12; // edx
  unsigned int v13; // r12d
  __int64 v15; // [rsp+0h] [rbp-70h] BYREF
  _BYTE *v16; // [rsp+8h] [rbp-68h]
  __int64 v17; // [rsp+10h] [rbp-60h]
  _BYTE v18[48]; // [rsp+18h] [rbp-58h] BYREF
  int v19; // [rsp+48h] [rbp-28h]

  v17 = 0x600000000LL;
  v15 = 0;
  v16 = v18;
  v19 = 0;
  sub_2ED4FB0((__int64)&v15, a3, a3, a4, a5, a6);
  sub_2E226F0(&v15, a1, v6, v7, v8, v9);
  v10 = *(_DWORD *)(*(_QWORD *)(v15 + 8) + 24LL * a2 + 16) & 0xFFF;
  v11 = (__int16 *)(*(_QWORD *)(v15 + 56) + 2LL * (*(_DWORD *)(*(_QWORD *)(v15 + 8) + 24LL * a2 + 16) >> 12));
  while ( 1 )
  {
    if ( !v11 )
    {
LABEL_5:
      v13 = 0;
      goto LABEL_6;
    }
    if ( (*(_QWORD *)&v16[8 * (v10 >> 6)] & (1LL << v10)) != 0 )
      break;
    v12 = *v11++;
    v10 += v12;
    if ( !(_WORD)v12 )
      goto LABEL_5;
  }
  v13 = 1;
LABEL_6:
  if ( v16 != v18 )
    _libc_free((unsigned __int64)v16);
  return v13;
}
