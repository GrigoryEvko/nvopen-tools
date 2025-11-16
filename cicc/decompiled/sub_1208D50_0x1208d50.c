// Function: sub_1208D50
// Address: 0x1208d50
//
__int64 __fastcall sub_1208D50(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r15
  unsigned __int64 v5; // rsi
  __int64 v9; // rax
  int v10; // eax
  __int64 *v11; // r8
  __int64 v12; // rcx
  unsigned __int64 v13; // rsi
  __int64 v14; // rax
  int v15; // eax
  unsigned int v16; // edx
  __int64 *v17; // rsi
  __int64 v18; // rax
  int v19; // [rsp+10h] [rbp-D0h]
  int v20; // [rsp+18h] [rbp-C8h]
  _QWORD v21[2]; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v22; // [rsp+30h] [rbp-B0h]
  __int64 v23; // [rsp+38h] [rbp-A8h]
  __int16 v24; // [rsp+40h] [rbp-A0h]
  _QWORD v25[2]; // [rsp+50h] [rbp-90h] BYREF
  const char *v26; // [rsp+60h] [rbp-80h]
  __int16 v27; // [rsp+70h] [rbp-70h]
  const char *v28; // [rsp+80h] [rbp-60h] BYREF
  unsigned int v29; // [rsp+88h] [rbp-58h]
  char v30; // [rsp+8Ch] [rbp-54h]
  __int64 v31; // [rsp+90h] [rbp-50h]
  __int16 v32; // [rsp+A0h] [rbp-40h]

  v4 = a1 + 176;
  if ( *(_DWORD *)(a1 + 240) != 529 )
  {
    v5 = *(_QWORD *)(a1 + 232);
    v32 = 259;
    v28 = "expected signed integer";
    sub_11FD800(a1 + 176, v5, (__int64)&v28, 1);
    return 1;
  }
  v9 = *(_QWORD *)(a4 + 16);
  v30 = 0;
  v29 = 64;
  v28 = (const char *)v9;
  v10 = sub_AA8A40((__int64 *)(a1 + 320), (__int64 *)&v28);
  v11 = (__int64 *)(a1 + 320);
  if ( v29 > 0x40 && v28 )
  {
    v19 = v10;
    j_j___libc_free_0_0(v28);
    v10 = v19;
    v11 = (__int64 *)(a1 + 320);
  }
  if ( v10 < 0 )
  {
    v22 = a2;
    v21[0] = "value for '";
    v12 = a4 + 16;
    v25[0] = v21;
    v26 = "' too small, limit is ";
    v24 = 1283;
    v23 = a3;
    v27 = 770;
    v28 = (const char *)v25;
LABEL_8:
    v13 = *(_QWORD *)(a1 + 232);
    v31 = v12;
    v32 = 3074;
    sub_11FD800(v4, v13, (__int64)&v28, 1);
    return 1;
  }
  v14 = *(_QWORD *)(a4 + 24);
  v30 = 0;
  v29 = 64;
  v28 = (const char *)v14;
  v15 = sub_AA8A40(v11, (__int64 *)&v28);
  if ( v29 > 0x40 && v28 )
  {
    v20 = v15;
    j_j___libc_free_0_0(v28);
    v15 = v20;
  }
  if ( v15 > 0 )
  {
    v22 = a2;
    v21[0] = "value for '";
    v25[0] = v21;
    v26 = "' too large, limit is ";
    v24 = 1283;
    v12 = a4 + 24;
    v23 = a3;
    v27 = 770;
    v28 = (const char *)v25;
    goto LABEL_8;
  }
  v16 = *(_DWORD *)(a1 + 328);
  v17 = *(__int64 **)(a1 + 320);
  if ( !*(_BYTE *)(a1 + 332) )
  {
    if ( v16 <= 0x40 )
    {
      v18 = 0;
      if ( v16 )
        v18 = (__int64)((_QWORD)v17 << (64 - (unsigned __int8)v16)) >> (64 - (unsigned __int8)v16);
      goto LABEL_20;
    }
    goto LABEL_19;
  }
  v18 = *(_QWORD *)(a1 + 320);
  if ( v16 > 0x40 )
LABEL_19:
    v18 = *v17;
LABEL_20:
  *(_BYTE *)(a4 + 8) = 1;
  *(_QWORD *)a4 = v18;
  *(_DWORD *)(a1 + 240) = sub_1205200(v4);
  return 0;
}
