// Function: sub_39457C0
// Address: 0x39457c0
//
void __fastcall sub_39457C0(__int64 a1, char *a2, unsigned __int64 a3)
{
  unsigned int v3; // r12d
  unsigned int v4; // r13d
  unsigned __int64 v5; // rax
  unsigned int v6; // edx
  unsigned __int64 v7; // r12
  bool v8; // cc
  int v9; // eax
  __int64 v10; // rax
  int v11; // eax
  unsigned int v12; // r15d
  bool v13; // al
  unsigned __int64 v14; // r12
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rsi
  unsigned __int64 v17; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v18; // [rsp+8h] [rbp-48h]
  unsigned __int64 v19; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v20; // [rsp+18h] [rbp-38h]

  *(_DWORD *)(a1 + 8) = 1;
  *(_QWORD *)a1 = 0;
  v3 = (a3 << 6) / 0x13 + 2;
  sub_16A9890((__int64)&v17, v3, a2, a3, 0xAu);
  v4 = v18;
  if ( *a2 != 45 )
  {
    if ( v18 > 0x40 )
    {
      v9 = sub_16A57B0((__int64)&v17);
      v6 = v4 - v9;
      if ( v4 == v9 || v3 <= v6 )
      {
        v20 = v4;
        goto LABEL_22;
      }
    }
    else
    {
      if ( !v17 )
        goto LABEL_5;
      _BitScanReverse64(&v5, v17);
      v6 = 64 - (v5 ^ 0x3F);
      if ( v3 <= v6 )
        goto LABEL_5;
    }
    sub_16A5A50((__int64)&v19, (__int64 *)&v17, v6);
    if ( v18 > 0x40 && v17 )
      j_j___libc_free_0_0(v17);
    v4 = v20;
    v17 = v19;
    v18 = v20;
    if ( v20 <= 0x40 )
    {
LABEL_5:
      v7 = v17;
      v8 = *(_DWORD *)(a1 + 8) <= 0x40u;
      v20 = 0;
      v19 = v17;
      if ( v8 )
        goto LABEL_23;
      goto LABEL_6;
    }
LABEL_22:
    sub_16A4FD0((__int64)&v19, (const void **)&v17);
    v8 = *(_DWORD *)(a1 + 8) <= 0x40u;
    v4 = v20;
    v20 = 0;
    v7 = v19;
    if ( v8 )
      goto LABEL_23;
LABEL_6:
    if ( *(_QWORD *)a1 )
    {
      j_j___libc_free_0_0(*(_QWORD *)a1);
      v8 = v20 <= 0x40;
      *(_QWORD *)a1 = v7;
      *(_DWORD *)(a1 + 8) = v4;
      *(_BYTE *)(a1 + 12) = 1;
      if ( v8 )
        goto LABEL_10;
LABEL_8:
      if ( v19 )
        j_j___libc_free_0_0(v19);
      goto LABEL_10;
    }
LABEL_23:
    *(_QWORD *)a1 = v7;
    *(_DWORD *)(a1 + 8) = v4;
    *(_BYTE *)(a1 + 12) = 1;
    goto LABEL_10;
  }
  v10 = 1LL << ((unsigned __int8)v18 - 1);
  if ( v18 > 0x40 )
  {
    if ( (*(_QWORD *)(v17 + 8LL * ((v18 - 1) >> 6)) & v10) != 0 )
      v11 = sub_16A5810((__int64)&v17);
    else
      v11 = sub_16A57B0((__int64)&v17);
LABEL_27:
    v12 = v4 + 1 - v11;
    v13 = v4 + 1 != v11;
    goto LABEL_28;
  }
  if ( (v17 & v10) != 0 )
  {
    v11 = 64;
    if ( v17 << (64 - (unsigned __int8)v18) != -1 )
    {
      _BitScanReverse64(&v15, ~(v17 << (64 - (unsigned __int8)v18)));
      v11 = v15 ^ 0x3F;
    }
    goto LABEL_27;
  }
  v12 = 1;
  if ( v17 )
  {
    _BitScanReverse64(&v16, v17);
    v12 = 65 - (v16 ^ 0x3F);
  }
  v13 = 1;
LABEL_28:
  if ( v3 > v12 && v13 )
  {
    sub_16A5A50((__int64)&v19, (__int64 *)&v17, v12);
    if ( v18 > 0x40 && v17 )
      j_j___libc_free_0_0(v17);
    v4 = v20;
    v17 = v19;
    v18 = v20;
  }
  v20 = v4;
  if ( v4 > 0x40 )
  {
    sub_16A4FD0((__int64)&v19, (const void **)&v17);
    v4 = v20;
    v14 = v19;
  }
  else
  {
    v14 = v17;
    v19 = v17;
  }
  v8 = *(_DWORD *)(a1 + 8) <= 0x40u;
  v20 = 0;
  if ( v8 || !*(_QWORD *)a1 )
  {
    *(_QWORD *)a1 = v14;
    *(_DWORD *)(a1 + 8) = v4;
    *(_BYTE *)(a1 + 12) = 0;
    goto LABEL_10;
  }
  j_j___libc_free_0_0(*(_QWORD *)a1);
  v8 = v20 <= 0x40;
  *(_QWORD *)a1 = v14;
  *(_DWORD *)(a1 + 8) = v4;
  *(_BYTE *)(a1 + 12) = 0;
  if ( !v8 )
    goto LABEL_8;
LABEL_10:
  if ( v18 > 0x40 )
  {
    if ( v17 )
      j_j___libc_free_0_0(v17);
  }
}
