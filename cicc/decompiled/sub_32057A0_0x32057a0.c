// Function: sub_32057A0
// Address: 0x32057a0
//
__int64 __fastcall sub_32057A0(_QWORD *a1, __int64 a2)
{
  __int16 v4; // ax
  __int16 v5; // bx
  __int16 v6; // r13
  unsigned __int8 v7; // al
  __int64 v8; // rsi
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // rax
  unsigned int v12; // r13d
  int v14; // [rsp+14h] [rbp-ACh]
  int v15; // [rsp+18h] [rbp-A8h]
  __int16 v16; // [rsp+1Ch] [rbp-A4h]
  unsigned __int64 v17[2]; // [rsp+30h] [rbp-90h] BYREF
  __int64 v18; // [rsp+40h] [rbp-80h] BYREF
  _WORD v19[2]; // [rsp+50h] [rbp-70h] BYREF
  __int16 v20; // [rsp+54h] [rbp-6Ch]
  _BYTE v21[6]; // [rsp+56h] [rbp-6Ah]
  int v22; // [rsp+5Ch] [rbp-64h]
  unsigned __int64 v23; // [rsp+60h] [rbp-60h]
  unsigned __int64 v24; // [rsp+68h] [rbp-58h]
  __int64 v25; // [rsp+70h] [rbp-50h]
  __int64 v26; // [rsp+78h] [rbp-48h]
  int v27; // [rsp+80h] [rbp-40h]
  int v28; // [rsp+84h] [rbp-3Ch]
  __int64 v29; // [rsp+88h] [rbp-38h]

  v4 = sub_AF18C0(a2);
  if ( v4 == 2 )
  {
    v5 = 5380;
  }
  else
  {
    if ( v4 != 19 )
      BUG();
    v5 = 5381;
  }
  v6 = sub_31F58C0(a2);
  sub_3204160((__int64)v19, (__int64)a1, a2);
  v15 = v22;
  v14 = *(_DWORD *)&v21[2];
  v16 = v20;
  if ( LOBYTE(v19[0]) )
    v6 |= 0x10u;
  if ( (*(_BYTE *)(a2 + 23) & 4) != 0 )
    v6 |= 2u;
  sub_3205740((__int64)v17, (__int64)a1, (unsigned __int8 *)a2);
  v7 = *(_BYTE *)(a2 - 16);
  v8 = *(_QWORD *)(a2 + 24) >> 3;
  if ( (v7 & 2) != 0 )
  {
    v9 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 56LL);
    if ( v9 )
    {
LABEL_9:
      v8 = *(_QWORD *)(a2 + 24) >> 3;
      v9 = sub_B91420(v9);
      goto LABEL_10;
    }
  }
  else
  {
    v9 = *(_QWORD *)(a2 - 16 - 8LL * ((v7 >> 2) & 0xF) + 56);
    if ( v9 )
      goto LABEL_9;
  }
  v10 = 0;
LABEL_10:
  v25 = v9;
  v29 = v8;
  v19[1] = v16;
  v20 = v6;
  *(_DWORD *)v21 = v15;
  v23 = v17[0];
  v19[0] = v5;
  v24 = v17[1];
  v26 = v10;
  v27 = 0;
  v28 = v14;
  v11 = sub_3709F10(a1 + 81, v19);
  v12 = sub_3707F80(a1 + 79, v11);
  sub_31FDA50(a1, a2, v12);
  sub_31FBCA0(a1, (unsigned __int64 *)a2);
  if ( (__int64 *)v17[0] != &v18 )
    j_j___libc_free_0(v17[0]);
  return v12;
}
