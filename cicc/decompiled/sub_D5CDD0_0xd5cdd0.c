// Function: sub_D5CDD0
// Address: 0xd5cdd0
//
__int64 __fastcall sub_D5CDD0(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        __int64 (__fastcall *a4)(__int64, _QWORD),
        __int64 a5)
{
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  unsigned int v13; // eax
  unsigned int v14; // eax
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rax
  unsigned int v18; // ebx
  __int64 v19; // r13
  const void *v20; // rax
  int v21; // [rsp+0h] [rbp-A0h]
  unsigned int v22; // [rsp+0h] [rbp-A0h]
  unsigned int v23; // [rsp+Ch] [rbp-94h]
  bool v24; // [rsp+1Fh] [rbp-81h] BYREF
  __int64 v25; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v26; // [rsp+28h] [rbp-78h]
  const void *v27; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v28; // [rsp+38h] [rbp-68h]
  const void *v29; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v30; // [rsp+48h] [rbp-58h]
  __m128i v31; // [rsp+50h] [rbp-50h] BYREF
  char v32; // [rsp+68h] [rbp-38h]

  sub_D5BE70(&v31, (unsigned __int8 *)a2, a3);
  if ( !v32 )
    goto LABEL_2;
  v9 = v31.m128i_i32[2];
  v21 = v31.m128i_i32[3];
  v10 = sub_B43CC0(a2);
  v23 = sub_AE43F0(v10, *(_QWORD *)(a2 + 8));
  if ( v31.m128i_i8[0] != 4 )
  {
    v11 = a4(a5, *(_QWORD *)(a2 + 32 * ((unsigned int)v9 - (unsigned __int64)(*(_DWORD *)(a2 + 4) & 0x7FFFFFF))));
    if ( *(_BYTE *)v11 != 17 )
    {
LABEL_2:
      *(_BYTE *)(a1 + 16) = 0;
      return a1;
    }
    v26 = *(_DWORD *)(v11 + 32);
    if ( v26 > 0x40 )
      sub_C43780((__int64)&v25, (const void **)(v11 + 24));
    else
      v25 = *(_QWORD *)(v11 + 24);
    if ( !(unsigned __int8)sub_D5C0A0(&v25, v23) )
      goto LABEL_29;
    if ( v21 >= 0 )
    {
      v12 = a4(a5, *(_QWORD *)(a2 + 32 * (v21 - (unsigned __int64)(*(_DWORD *)(a2 + 4) & 0x7FFFFFF))));
      if ( *(_BYTE *)v12 == 17 )
      {
        v28 = *(_DWORD *)(v12 + 32);
        if ( v28 > 0x40 )
          sub_C43780((__int64)&v27, (const void **)(v12 + 24));
        else
          v27 = *(const void **)(v12 + 24);
        if ( !(unsigned __int8)sub_D5C0A0(&v27, v23) )
          goto LABEL_14;
        sub_C49BE0((__int64)&v29, (__int64)&v25, (__int64)&v27, &v24);
        if ( v26 > 0x40 && v25 )
          j_j___libc_free_0_0(v25);
        v20 = v29;
        v25 = (__int64)v29;
        v26 = v30;
        if ( v24 )
        {
LABEL_14:
          *(_BYTE *)(a1 + 16) = 0;
        }
        else
        {
          *(_DWORD *)(a1 + 8) = v30;
          *(_QWORD *)a1 = v20;
          v26 = 0;
          *(_BYTE *)(a1 + 16) = 1;
        }
        if ( v28 > 0x40 && v27 )
          j_j___libc_free_0_0(v27);
LABEL_30:
        if ( v26 <= 0x40 )
          return a1;
        goto LABEL_31;
      }
LABEL_29:
      *(_BYTE *)(a1 + 16) = 0;
      goto LABEL_30;
    }
LABEL_21:
    v14 = v26;
    *(_BYTE *)(a1 + 16) = 1;
    *(_DWORD *)(a1 + 8) = v14;
    *(_QWORD *)a1 = v25;
    return a1;
  }
  v15 = a4(a5, *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  v16 = sub_98B430(v15, 8u);
  v26 = v23;
  if ( v23 <= 0x40 )
  {
    v25 = v16;
    goto LABEL_24;
  }
  sub_C43690((__int64)&v25, v16, 0);
  if ( v26 <= 0x40 )
  {
    v16 = v25;
LABEL_24:
    if ( !v16 )
    {
      v13 = v26;
      goto LABEL_26;
    }
    goto LABEL_20;
  }
  v22 = v26;
  v13 = sub_C444A0((__int64)&v25);
  if ( v22 == v13 )
  {
LABEL_26:
    *(_BYTE *)(a1 + 16) = 0;
    goto LABEL_27;
  }
LABEL_20:
  if ( (int)v9 <= 0 )
    goto LABEL_21;
  v17 = a4(a5, *(_QWORD *)(a2 + 32 * (v9 - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))));
  if ( *(_BYTE *)v17 == 17 )
  {
    sub_C449B0((__int64)&v27, (const void **)(v17 + 24), v23);
    if ( (int)sub_C49970((__int64)&v25, (unsigned __int64 *)&v27) > 0 )
    {
      v30 = v28;
      if ( v28 > 0x40 )
        sub_C43780((__int64)&v29, &v27);
      else
        v29 = v27;
      sub_C46A40((__int64)&v29, 1);
      v18 = v30;
      v30 = 0;
      v19 = (__int64)v29;
      if ( v26 > 0x40 && v25 )
      {
        j_j___libc_free_0_0(v25);
        v25 = v19;
        v26 = v18;
        if ( v30 > 0x40 && v29 )
          j_j___libc_free_0_0(v29);
      }
      else
      {
        v25 = (__int64)v29;
        v26 = v18;
      }
    }
    if ( v28 > 0x40 && v27 )
      j_j___libc_free_0_0(v27);
    goto LABEL_21;
  }
  *(_BYTE *)(a1 + 16) = 0;
  v13 = v26;
LABEL_27:
  if ( v13 <= 0x40 )
    return a1;
LABEL_31:
  if ( v25 )
    j_j___libc_free_0_0(v25);
  return a1;
}
