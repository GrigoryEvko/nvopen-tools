// Function: sub_14AA5A0
// Address: 0x14aa5a0
//
__int64 __fastcall sub_14AA5A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v6; // eax
  __int64 v7; // r12
  unsigned __int64 v8; // rdx
  unsigned int v9; // ecx
  __int64 result; // rax
  unsigned __int64 v11; // rdx
  __int64 v12; // r13
  __int64 v13; // rsi
  unsigned int v14; // edx
  __int64 v15; // rax
  bool v16; // cc
  unsigned int v17; // edx
  unsigned __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rdx
  __int64 v21; // rsi
  unsigned int v22; // eax
  unsigned __int64 v23; // r8
  unsigned int v24; // edx
  unsigned __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rdi
  __int64 v28; // rcx
  __int64 v29; // rax
  __int64 v30; // rcx
  unsigned int v31; // [rsp+10h] [rbp-C0h]
  unsigned int v32; // [rsp+10h] [rbp-C0h]
  unsigned int v33; // [rsp+10h] [rbp-C0h]
  __int64 v34; // [rsp+10h] [rbp-C0h]
  unsigned int v35; // [rsp+10h] [rbp-C0h]
  __int64 v36; // [rsp+10h] [rbp-C0h]
  unsigned int v38; // [rsp+2Ch] [rbp-A4h]
  __int64 v39; // [rsp+40h] [rbp-90h] BYREF
  unsigned int v40; // [rsp+48h] [rbp-88h]
  unsigned __int64 v41; // [rsp+50h] [rbp-80h] BYREF
  unsigned int v42; // [rsp+58h] [rbp-78h]
  __int64 v43; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v44; // [rsp+68h] [rbp-68h]
  __int64 v45; // [rsp+70h] [rbp-60h] BYREF
  unsigned int v46; // [rsp+78h] [rbp-58h]
  __int64 v47; // [rsp+80h] [rbp-50h] BYREF
  unsigned int v48; // [rsp+88h] [rbp-48h]
  __int64 v49; // [rsp+90h] [rbp-40h]
  unsigned int v50; // [rsp+98h] [rbp-38h]

  v6 = *(_DWORD *)(a2 + 8);
  v38 = v6;
  v7 = *(_DWORD *)(a1 + 8) >> 1;
  if ( v6 <= 0x40 )
  {
    *(_QWORD *)a2 = -1;
    v8 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v6;
LABEL_3:
    *(_QWORD *)a2 &= v8;
    goto LABEL_4;
  }
  memset(*(void **)a2, -1, 8 * (((unsigned __int64)v6 + 63) >> 6));
  v29 = *(unsigned int *)(a2 + 8);
  v8 = 0xFFFFFFFFFFFFFFFFLL >> -*(_BYTE *)(a2 + 8);
  if ( (unsigned int)v29 <= 0x40 )
    goto LABEL_3;
  v30 = (unsigned int)((unsigned __int64)(v29 + 63) >> 6) - 1;
  *(_QWORD *)(*(_QWORD *)a2 + 8 * v30) &= v8;
LABEL_4:
  v9 = *(_DWORD *)(a2 + 24);
  result = a2 + 16;
  if ( v9 <= 0x40 )
  {
    *(_QWORD *)(a2 + 16) = -1;
    v11 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v9;
LABEL_6:
    *(_QWORD *)(a2 + 16) &= v11;
    goto LABEL_7;
  }
  memset(*(void **)(a2 + 16), -1, 8 * (((unsigned __int64)v9 + 63) >> 6));
  result = *(unsigned int *)(a2 + 24);
  v11 = 0xFFFFFFFFFFFFFFFFLL >> -*(_BYTE *)(a2 + 24);
  if ( (unsigned int)result <= 0x40 )
    goto LABEL_6;
  v28 = (unsigned int)((unsigned __int64)(result + 63) >> 6) - 1;
  result = *(_QWORD *)(a2 + 16);
  *(_QWORD *)(result + 8 * v28) &= v11;
LABEL_7:
  if ( (_DWORD)v7 )
  {
    v12 = 0;
    while ( 1 )
    {
      v19 = *(unsigned int *)(a1 + 8);
      v20 = *(_QWORD *)(*(_QWORD *)(a1 + 8 * (v12 - v19)) + 136LL);
      v21 = *(_QWORD *)(*(_QWORD *)(a1 + 8 * (v12 + 1 - v19)) + 136LL);
      v46 = *(_DWORD *)(v21 + 32);
      if ( v46 > 0x40 )
      {
        v36 = v20;
        sub_16A4FD0(&v45, v21 + 24);
        v20 = v36;
      }
      else
      {
        v45 = *(_QWORD *)(v21 + 24);
      }
      v44 = *(_DWORD *)(v20 + 32);
      if ( v44 > 0x40 )
        sub_16A4FD0(&v43, v20 + 24);
      else
        v43 = *(_QWORD *)(v20 + 24);
      sub_15898E0(&v47, &v43, &v45, v19, a5);
      if ( v44 > 0x40 && v43 )
        j_j___libc_free_0_0(v43);
      if ( v46 > 0x40 && v45 )
        j_j___libc_free_0_0(v45);
      sub_158AAD0(&v43, &v47);
      sub_158A9F0(&v41, &v47);
      v22 = v44;
      if ( v44 <= 0x40 )
        break;
      sub_16A8F00(&v43, &v41);
      v22 = v44;
      v23 = v43;
      v44 = 0;
      v46 = v22;
      v45 = v43;
      if ( v22 <= 0x40 )
        goto LABEL_60;
      v34 = v43;
      v22 = sub_16A57B0(&v45);
      if ( v34 )
      {
        v27 = v34;
        v35 = v22;
        j_j___libc_free_0_0(v27);
        v22 = v35;
      }
LABEL_62:
      if ( v42 > 0x40 && v41 )
      {
        v31 = v22;
        j_j___libc_free_0_0(v41);
        v22 = v31;
      }
      if ( v44 > 0x40 && v43 )
      {
        v32 = v22;
        j_j___libc_free_0_0(v43);
        v22 = v32;
      }
      v26 = v38;
      v40 = v38;
      if ( v38 <= 0x40 )
      {
        v39 = 0;
      }
      else
      {
        v33 = v22;
        sub_16A4EF0(&v39, 0, 0);
        v26 = v40;
        v22 = v33;
      }
      v13 = (unsigned int)v26 - v22;
      if ( (_DWORD)v13 != (_DWORD)v26 )
      {
        if ( (unsigned int)v13 > 0x3F || (unsigned int)v26 > 0x40 )
          sub_16A5260(&v39, v13, v26);
        else
          v39 |= 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v22) << ((unsigned __int8)v26 - (unsigned __int8)v22);
      }
      sub_158A9F0(&v43, &v47);
      v14 = v44;
      if ( v44 > 0x40 )
      {
        sub_16A8890(&v43, &v39);
        v14 = v44;
        v15 = v43;
      }
      else
      {
        v15 = v39 & v43;
        v43 &= v39;
      }
      v16 = *(_DWORD *)(a2 + 24) <= 0x40u;
      v46 = v14;
      v45 = v15;
      v44 = 0;
      if ( v16 )
      {
        *(_QWORD *)(a2 + 16) &= v15;
      }
      else
      {
        sub_16A8890(a2 + 16, &v45);
        v14 = v46;
      }
      if ( v14 > 0x40 && v45 )
        j_j___libc_free_0_0(v45);
      if ( v44 > 0x40 && v43 )
        j_j___libc_free_0_0(v43);
      sub_158A9F0(&v41, &v47);
      v17 = v42;
      if ( v42 > 0x40 )
      {
        sub_16A8F40(&v41);
        v17 = v42;
        v18 = v41;
        v42 = 0;
        v44 = v17;
        v43 = v41;
        if ( v17 > 0x40 )
        {
          sub_16A8890(&v43, &v39);
          result = v43;
          v17 = v44;
          goto LABEL_27;
        }
      }
      else
      {
        v42 = 0;
        v18 = ~v41 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v17);
        v41 = v18;
      }
      result = v39 & v18;
      v43 = result;
LABEL_27:
      v16 = *(_DWORD *)(a2 + 8) <= 0x40u;
      v46 = v17;
      v45 = result;
      v44 = 0;
      if ( v16 )
      {
        *(_QWORD *)a2 &= result;
      }
      else
      {
        result = sub_16A8890(a2, &v45);
        v17 = v46;
      }
      if ( v17 > 0x40 && v45 )
        result = j_j___libc_free_0_0(v45);
      if ( v44 > 0x40 && v43 )
        result = j_j___libc_free_0_0(v43);
      if ( v42 > 0x40 && v41 )
        result = j_j___libc_free_0_0(v41);
      if ( v40 > 0x40 && v39 )
        result = j_j___libc_free_0_0(v39);
      if ( v50 > 0x40 && v49 )
        result = j_j___libc_free_0_0(v49);
      if ( v48 > 0x40 )
      {
        if ( v47 )
          result = j_j___libc_free_0_0(v47);
      }
      v12 += 2;
      if ( v12 == 2 * v7 )
        return result;
    }
    v23 = v41 ^ v43;
    v44 = 0;
    v43 ^= v41;
LABEL_60:
    v24 = v22 - 64;
    if ( v23 )
    {
      _BitScanReverse64(&v25, v23);
      v22 = v24 + (v25 ^ 0x3F);
    }
    goto LABEL_62;
  }
  return result;
}
