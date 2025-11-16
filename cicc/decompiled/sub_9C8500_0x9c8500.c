// Function: sub_9C8500
// Address: 0x9c8500
//
__int64 __fastcall sub_9C8500(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int *a5, unsigned int a6)
{
  __int64 v8; // rdx
  int v10; // eax
  __int64 v11; // r8
  unsigned __int64 *v13; // rdx
  unsigned __int64 v14; // r15
  __int64 v15; // r15
  bool v16; // cc
  __int64 v17; // rdi
  __int64 v18; // rax
  unsigned __int64 v20; // rcx
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // rcx
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // rsi
  unsigned __int64 v25; // rdx
  unsigned __int64 v26; // [rsp+8h] [rbp-A8h]
  __int64 v27; // [rsp+10h] [rbp-A0h] BYREF
  unsigned int v28; // [rsp+18h] [rbp-98h]
  __int64 v29; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v30; // [rsp+28h] [rbp-88h]
  unsigned __int64 v31; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v32; // [rsp+38h] [rbp-78h]
  __int64 v33; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v34; // [rsp+48h] [rbp-68h]
  const char *v35; // [rsp+50h] [rbp-60h] BYREF
  int v36; // [rsp+58h] [rbp-58h]
  __int64 v37; // [rsp+60h] [rbp-50h]
  int v38; // [rsp+68h] [rbp-48h]
  char v39; // [rsp+70h] [rbp-40h]
  char v40; // [rsp+71h] [rbp-3Fh]

  v8 = *a5;
  if ( (unsigned __int64)(a4 - v8) <= 1 )
    goto LABEL_20;
  v10 = *a5;
  v11 = (unsigned int)(v8 + 1);
  v13 = (unsigned __int64 *)(a3 + 8 * v8);
  if ( a6 <= 0x40 )
  {
    *a5 = v11;
    v20 = *v13;
    if ( (*v13 & 1) != 0 )
    {
      v21 = 0x8000000000000000LL;
      if ( v20 != 1 )
        v21 = -(__int64)(v20 >> 1);
    }
    else
    {
      v21 = v20 >> 1;
    }
    *a5 = v10 + 2;
    v22 = *(_QWORD *)(a3 + 8 * v11);
    v23 = v22 >> 1;
    if ( (v22 & 1) != 0 )
    {
      v23 = 0x8000000000000000LL;
      if ( v22 != 1 )
        v23 = -(__int64)(v22 >> 1);
    }
    v34 = a6;
    v24 = 0xFFFFFFFFFFFFFFFFLL >> -(char)a6;
    if ( a6 )
    {
      v32 = a6;
      v25 = v24 & v21;
      v33 = v24 & v23;
    }
    else
    {
      v33 = 0;
      v25 = 0;
      v32 = 0;
    }
    v31 = v25;
    sub_AADC30(&v35, &v31, &v33);
    v16 = v32 <= 0x40;
    *(_BYTE *)(a1 + 32) = *(_BYTE *)(a1 + 32) & 0xFC | 2;
    *(_DWORD *)(a1 + 8) = v36;
    *(_QWORD *)a1 = v35;
    *(_DWORD *)(a1 + 24) = v38;
    *(_QWORD *)(a1 + 16) = v37;
    if ( !v16 && v31 )
      j_j___libc_free_0_0(v31);
    if ( v34 > 0x40 )
    {
      v17 = v33;
      if ( v33 )
        goto LABEL_19;
    }
    return a1;
  }
  v14 = *v13;
  *a5 = v11;
  if ( a4 - v11 < (unsigned __int64)(unsigned int)(HIDWORD(*v13) + v14) )
  {
LABEL_20:
    v40 = 1;
    v39 = 3;
    v35 = "Too few records for range";
    sub_9C81F0(&v33, a2 + 8, (__int64)&v35);
    v18 = v33;
    *(_BYTE *)(a1 + 32) |= 3u;
    *(_QWORD *)a1 = v18 & 0xFFFFFFFFFFFFFFFELL;
    return a1;
  }
  v26 = HIDWORD(*v13);
  sub_9C7EB0((__int64)&v27, a3 + 8 * v11, (unsigned int)v14, a6);
  v15 = *a5 + (unsigned int)v14;
  *a5 = v15;
  sub_9C7EB0((__int64)&v29, a3 + 8 * v15, v26, a6);
  *a5 += v26;
  v34 = v30;
  if ( v30 > 0x40 )
    sub_C43780(&v33, &v29);
  else
    v33 = v29;
  v32 = v28;
  if ( v28 > 0x40 )
    sub_C43780(&v31, &v27);
  else
    v31 = v27;
  sub_AADC30(&v35, &v31, &v33);
  v16 = v32 <= 0x40;
  *(_BYTE *)(a1 + 32) = *(_BYTE *)(a1 + 32) & 0xFC | 2;
  *(_DWORD *)(a1 + 8) = v36;
  *(_QWORD *)a1 = v35;
  *(_DWORD *)(a1 + 24) = v38;
  *(_QWORD *)(a1 + 16) = v37;
  if ( !v16 && v31 )
    j_j___libc_free_0_0(v31);
  if ( v34 > 0x40 && v33 )
    j_j___libc_free_0_0(v33);
  if ( v30 > 0x40 && v29 )
    j_j___libc_free_0_0(v29);
  if ( v28 > 0x40 )
  {
    v17 = v27;
    if ( v27 )
LABEL_19:
      j_j___libc_free_0_0(v17);
  }
  return a1;
}
