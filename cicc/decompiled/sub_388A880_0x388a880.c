// Function: sub_388A880
// Address: 0x388a880
//
__int64 __fastcall sub_388A880(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r13
  bool v5; // zf
  unsigned __int64 v6; // rsi
  unsigned __int64 v8; // rax
  int v10; // eax
  __int64 v11; // rcx
  unsigned __int64 v12; // rsi
  unsigned __int64 v13; // rax
  int v14; // r14d
  unsigned int v15; // edx
  __int64 v16; // rax
  int v17; // [rsp+Ch] [rbp-A4h]
  _QWORD v18[2]; // [rsp+10h] [rbp-A0h] BYREF
  const char *v19; // [rsp+20h] [rbp-90h] BYREF
  _QWORD *v20; // [rsp+28h] [rbp-88h]
  __int16 v21; // [rsp+30h] [rbp-80h]
  const char **v22; // [rsp+40h] [rbp-70h] BYREF
  const char *v23; // [rsp+48h] [rbp-68h]
  __int16 v24; // [rsp+50h] [rbp-60h]
  const char *v25; // [rsp+60h] [rbp-50h] BYREF
  __int64 v26; // [rsp+68h] [rbp-48h]
  __int16 v27; // [rsp+70h] [rbp-40h]

  v4 = a1 + 8;
  v5 = *(_DWORD *)(a1 + 64) == 390;
  v18[0] = a2;
  v18[1] = a3;
  if ( !v5 )
  {
    v6 = *(_QWORD *)(a1 + 56);
    v27 = 259;
    v25 = "expected signed integer";
    return sub_38814C0(a1 + 8, v6, (__int64)&v25);
  }
  v8 = *(_QWORD *)(a4 + 16);
  BYTE4(v26) = 0;
  LODWORD(v26) = 64;
  v25 = (const char *)v8;
  v10 = sub_388A6C0(a1 + 152, (__int64)&v25);
  if ( (unsigned int)v26 > 0x40 && v25 )
  {
    v17 = v10;
    j_j___libc_free_0_0((unsigned __int64)v25);
    v10 = v17;
  }
  if ( v10 < 0 )
  {
    v11 = a4 + 16;
    v21 = 1283;
    v19 = "value for '";
    v20 = v18;
    v22 = &v19;
    v23 = "' too small, limit is ";
    v24 = 770;
    v25 = (const char *)&v22;
LABEL_8:
    v12 = *(_QWORD *)(a1 + 56);
    v26 = v11;
    v27 = 3074;
    return sub_38814C0(v4, v12, (__int64)&v25);
  }
  v13 = *(_QWORD *)(a4 + 24);
  BYTE4(v26) = 0;
  LODWORD(v26) = 64;
  v25 = (const char *)v13;
  v14 = sub_388A6C0(a1 + 152, (__int64)&v25);
  if ( (unsigned int)v26 > 0x40 )
  {
    if ( v25 )
      j_j___libc_free_0_0((unsigned __int64)v25);
  }
  if ( v14 > 0 )
  {
    v19 = "value for '";
    v20 = v18;
    v22 = &v19;
    v23 = "' too large, limit is ";
    v24 = 770;
    v11 = a4 + 24;
    v21 = 1283;
    v25 = (const char *)&v22;
    goto LABEL_8;
  }
  v15 = *(_DWORD *)(a1 + 160);
  v16 = *(_QWORD *)(a1 + 152);
  if ( !*(_BYTE *)(a1 + 164) )
  {
    if ( v15 <= 0x40 )
    {
      v16 = v16 << (64 - (unsigned __int8)v15) >> (64 - (unsigned __int8)v15);
      goto LABEL_17;
    }
    goto LABEL_16;
  }
  if ( v15 > 0x40 )
LABEL_16:
    v16 = *(_QWORD *)v16;
LABEL_17:
  *(_BYTE *)(a4 + 8) = 1;
  *(_QWORD *)a4 = v16;
  *(_DWORD *)(a1 + 64) = sub_3887100(v4);
  return 0;
}
