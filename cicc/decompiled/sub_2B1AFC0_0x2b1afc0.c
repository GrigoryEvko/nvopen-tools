// Function: sub_2B1AFC0
// Address: 0x2b1afc0
//
__int64 __fastcall sub_2B1AFC0(_DWORD **a1, __int64 a2)
{
  unsigned int v2; // r14d
  __int64 v3; // rdx
  unsigned int v4; // r12d
  unsigned int v5; // edx
  unsigned __int32 v6; // r12d
  __int64 v7; // rax
  unsigned __int64 v8; // rax
  unsigned int *v9; // rdx
  unsigned __int64 v10; // rdx
  unsigned __int64 *v11; // r15
  unsigned __int64 v12; // rax
  __int64 *v13; // rbx
  unsigned int *v15; // rax
  unsigned __int64 v16; // [rsp+8h] [rbp-C8h]
  unsigned __int64 v17; // [rsp+10h] [rbp-C0h] BYREF
  unsigned int v18; // [rsp+18h] [rbp-B8h]
  __int64 v19; // [rsp+20h] [rbp-B0h] BYREF
  __int32 v20; // [rsp+28h] [rbp-A8h]
  unsigned __int64 v21; // [rsp+30h] [rbp-A0h] BYREF
  unsigned __int32 v22; // [rsp+38h] [rbp-98h]
  unsigned __int64 v23; // [rsp+40h] [rbp-90h]
  unsigned int v24; // [rsp+48h] [rbp-88h]
  __m128i v25; // [rsp+50h] [rbp-80h] BYREF
  __int64 v26; // [rsp+60h] [rbp-70h]
  __int64 v27; // [rsp+68h] [rbp-68h]
  __int64 v28; // [rsp+70h] [rbp-60h]
  __int64 v29; // [rsp+78h] [rbp-58h]
  __int64 v30; // [rsp+80h] [rbp-50h]
  __int64 v31; // [rsp+88h] [rbp-48h]
  __int16 v32; // [rsp+90h] [rbp-40h]

  v2 = 1;
  if ( *(_BYTE *)a2 == 13 )
    return v2;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v3 = *(_QWORD *)(a2 - 8);
  else
    v3 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  sub_9AC3E0((__int64)&v21, *(_QWORD *)(v3 + 32), *((_QWORD *)*a1 + 418), 0, 0, 0, 0, 1);
  v4 = *a1[2];
  v5 = *a1[1];
  v18 = v5;
  if ( v5 > 0x40 )
  {
    sub_C43690((__int64)&v17, 0, 0);
    v5 = v18;
  }
  else
  {
    v17 = 0;
  }
  if ( v4 != v5 )
  {
    if ( v4 > 0x3F || v5 > 0x40 )
      sub_C43C90(&v17, v4, v5);
    else
      v17 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v4 + 64 - (unsigned __int8)v5) << v4;
  }
  v6 = v22;
  v25.m128i_i32[2] = v22;
  if ( v22 <= 0x40 )
  {
    v7 = v21;
LABEL_12:
    v20 = v6;
    v8 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v6) & ~v7;
    if ( !v6 )
      v8 = 0;
    v9 = a1[2];
    v19 = v8;
    v10 = *v9;
    v11 = (unsigned __int64 *)v8;
LABEL_15:
    v2 = 0;
    if ( v8 < v10 )
    {
      v12 = *((_QWORD *)*a1 + 418);
      v26 = 0;
      v25 = (__m128i)v12;
      v27 = 0;
      v28 = 0;
      v29 = 0;
      v30 = 0;
      v31 = 0;
      v32 = 257;
      if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
        v13 = *(__int64 **)(a2 - 8);
      else
        v13 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
      v2 = sub_9AC230(*v13, (__int64)&v17, &v25, 0);
    }
    if ( v6 <= 0x40 )
      goto LABEL_20;
    goto LABEL_35;
  }
  sub_C43780((__int64)&v25, (const void **)&v21);
  v6 = v25.m128i_u32[2];
  if ( v25.m128i_i32[2] <= 0x40u )
  {
    v7 = v25.m128i_i64[0];
    goto LABEL_12;
  }
  sub_C43D10((__int64)&v25);
  v6 = v25.m128i_u32[2];
  v11 = (unsigned __int64 *)v25.m128i_i64[0];
  v15 = a1[2];
  v20 = v25.m128i_i32[2];
  v19 = v25.m128i_i64[0];
  v10 = *v15;
  if ( v25.m128i_i32[2] <= 0x40u )
  {
    v8 = v25.m128i_i64[0];
    goto LABEL_15;
  }
  v16 = *v15;
  if ( v6 - (unsigned int)sub_C444A0((__int64)&v19) <= 0x40 )
  {
    v8 = *v11;
    v10 = v16;
    goto LABEL_15;
  }
  v2 = 0;
LABEL_35:
  if ( v11 )
    j_j___libc_free_0_0((unsigned __int64)v11);
LABEL_20:
  if ( v18 > 0x40 && v17 )
    j_j___libc_free_0_0(v17);
  if ( v24 > 0x40 && v23 )
    j_j___libc_free_0_0(v23);
  if ( v22 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
  return v2;
}
