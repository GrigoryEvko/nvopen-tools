// Function: sub_988CD0
// Address: 0x988cd0
//
__int64 __fastcall sub_988CD0(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int v5; // eax
  unsigned int v6; // ecx
  unsigned int v8; // ecx
  __int64 v9; // rdi
  __int64 v10; // rdi
  __int64 v11; // rsi
  unsigned int v12; // eax
  __int64 v13; // rsi
  __int64 v14; // [rsp+0h] [rbp-70h] BYREF
  __int64 v15; // [rsp+8h] [rbp-68h]
  __int64 v16; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v17; // [rsp+18h] [rbp-58h]
  __int64 v18; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v19; // [rsp+28h] [rbp-48h]
  __int64 v20; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v21; // [rsp+38h] [rbp-38h]
  __int64 v22; // [rsp+40h] [rbp-30h] BYREF
  unsigned int v23; // [rsp+48h] [rbp-28h]

  v14 = sub_B2D7D0(a2, 96);
  if ( !v14 )
  {
    v23 = a3;
    if ( a3 > 0x40 )
    {
      sub_C43690(&v22, 0, 0);
      v21 = a3;
      sub_C43690(&v20, 1, 0);
    }
    else
    {
      v22 = 0;
      v20 = 1;
      v21 = a3;
    }
    sub_AADC30(a1, &v20, &v22);
    if ( v21 > 0x40 && v20 )
      j_j___libc_free_0_0(v20);
    if ( v23 <= 0x40 )
      return a1;
    v10 = v22;
    if ( !v22 )
      return a1;
    goto LABEL_21;
  }
  v5 = sub_A71EB0(&v14);
  if ( v5 )
  {
    _BitScanReverse(&v6, v5);
    if ( 32 - (v6 ^ 0x1F) > a3 )
    {
      sub_AADB10(a1, a3, 0);
      return a1;
    }
  }
  v17 = a3;
  if ( a3 <= 0x40 )
  {
    v16 = v5;
    v15 = sub_A71ED0(&v14);
    if ( !BYTE4(v15) )
    {
LABEL_10:
      v20 = 0;
      v21 = a3;
      goto LABEL_11;
    }
    if ( (_DWORD)v15 )
    {
      _BitScanReverse(&v8, v15);
      if ( a3 < 32 - (v8 ^ 0x1F) )
        goto LABEL_10;
      v19 = a3;
      v13 = (unsigned int)v15;
    }
    else
    {
      v19 = a3;
      v13 = 0;
    }
    v18 = v13;
LABEL_35:
    sub_C46A40(&v18, 1);
    v12 = v19;
    v19 = 0;
    v21 = v12;
    v20 = v18;
    v23 = v17;
    if ( v17 > 0x40 )
      sub_C43780(&v22, &v16);
    else
      v22 = v16;
    sub_AADC30(a1, &v22, &v20);
    if ( v23 > 0x40 && v22 )
      j_j___libc_free_0_0(v22);
    if ( v21 > 0x40 && v20 )
      j_j___libc_free_0_0(v20);
    if ( v19 > 0x40 )
    {
      v9 = v18;
      if ( v18 )
        goto LABEL_18;
    }
    goto LABEL_19;
  }
  sub_C43690(&v16, v5, 0);
  v15 = sub_A71ED0(&v14);
  if ( BYTE4(v15) )
  {
    v19 = a3;
    if ( (_DWORD)v15 )
      v11 = (unsigned int)v15;
    else
      v11 = 0;
    sub_C43690(&v18, v11, 0);
    goto LABEL_35;
  }
  v21 = a3;
  sub_C43690(&v20, 0, 0);
LABEL_11:
  v23 = v17;
  if ( v17 > 0x40 )
    sub_C43780(&v22, &v16);
  else
    v22 = v16;
  sub_AADC30(a1, &v22, &v20);
  if ( v23 > 0x40 && v22 )
    j_j___libc_free_0_0(v22);
  if ( v21 > 0x40 )
  {
    v9 = v20;
    if ( v20 )
LABEL_18:
      j_j___libc_free_0_0(v9);
  }
LABEL_19:
  if ( v17 <= 0x40 )
    return a1;
  v10 = v16;
  if ( !v16 )
    return a1;
LABEL_21:
  j_j___libc_free_0_0(v10);
  return a1;
}
