// Function: sub_9884E0
// Address: 0x9884e0
//
__int64 __fastcall sub_9884E0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  int v4; // r12d
  __int64 v5; // rsi
  unsigned __int64 v6; // rdx
  __int64 v7; // rax
  __int64 result; // rax
  __int64 v9; // rsi
  unsigned __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // rax
  __int64 v14; // rsi
  unsigned int v15; // eax
  unsigned __int64 v16; // r8
  unsigned int v17; // edx
  unsigned __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rsi
  unsigned int v21; // eax
  __int64 v22; // rdx
  __int64 v23; // rdx
  bool v24; // cc
  unsigned int v25; // edx
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rax
  unsigned __int8 v28; // al
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // rdi
  unsigned int v33; // [rsp+10h] [rbp-D0h]
  unsigned int v34; // [rsp+10h] [rbp-D0h]
  __int64 v35; // [rsp+10h] [rbp-D0h]
  unsigned int v36; // [rsp+10h] [rbp-D0h]
  unsigned int v37; // [rsp+10h] [rbp-D0h]
  __int64 v38; // [rsp+10h] [rbp-D0h]
  __int64 v39; // [rsp+18h] [rbp-C8h]
  unsigned int v41; // [rsp+34h] [rbp-ACh]
  __int64 v42; // [rsp+40h] [rbp-A0h] BYREF
  unsigned int v43; // [rsp+48h] [rbp-98h]
  unsigned __int64 v44; // [rsp+50h] [rbp-90h] BYREF
  unsigned int v45; // [rsp+58h] [rbp-88h]
  unsigned __int64 v46; // [rsp+60h] [rbp-80h] BYREF
  unsigned int v47; // [rsp+68h] [rbp-78h]
  __int64 v48; // [rsp+70h] [rbp-70h] BYREF
  unsigned int v49; // [rsp+78h] [rbp-68h]
  __int64 v50; // [rsp+80h] [rbp-60h] BYREF
  unsigned int v51; // [rsp+88h] [rbp-58h]
  __int64 v52; // [rsp+90h] [rbp-50h] BYREF
  unsigned int v53; // [rsp+98h] [rbp-48h]
  __int64 v54; // [rsp+A0h] [rbp-40h]
  unsigned int v55; // [rsp+A8h] [rbp-38h]

  v41 = *(_DWORD *)(a2 + 8);
  if ( (*(_BYTE *)(a1 - 16) & 2) != 0 )
  {
    v3 = v41;
    v4 = *(_DWORD *)(a1 - 24) >> 1;
    if ( v41 <= 0x40 )
    {
LABEL_3:
      *(_QWORD *)a2 = -1;
      v5 = -1;
      goto LABEL_4;
    }
  }
  else
  {
    v3 = v41;
    v4 = ((*(_WORD *)(a1 - 16) >> 6) & 0xFu) >> 1;
    if ( v41 <= 0x40 )
      goto LABEL_3;
  }
  memset(*(void **)a2, -1, 8 * (((unsigned __int64)v41 + 63) >> 6));
  v3 = *(unsigned int *)(a2 + 8);
  v5 = *(_QWORD *)a2;
LABEL_4:
  v6 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v3;
  if ( (_DWORD)v3 )
  {
    if ( (unsigned int)v3 > 0x40 )
    {
      v7 = (unsigned int)((unsigned __int64)(v3 + 63) >> 6) - 1;
      *(_QWORD *)(v5 + 8 * v7) &= v6;
      goto LABEL_7;
    }
  }
  else
  {
    v6 = 0;
  }
  *(_QWORD *)a2 = v5 & v6;
LABEL_7:
  result = *(unsigned int *)(a2 + 24);
  if ( (unsigned int)result > 0x40 )
  {
    memset(*(void **)(a2 + 16), -1, 8 * (((unsigned __int64)(unsigned int)result + 63) >> 6));
    result = *(unsigned int *)(a2 + 24);
    v9 = *(_QWORD *)(a2 + 16);
  }
  else
  {
    *(_QWORD *)(a2 + 16) = -1;
    v9 = -1;
  }
  v10 = 0xFFFFFFFFFFFFFFFFLL >> -(char)result;
  if ( (_DWORD)result )
  {
    if ( (unsigned int)result > 0x40 )
    {
      result = (unsigned int)((unsigned __int64)(result + 63) >> 6) - 1;
      *(_QWORD *)(v9 + 8 * result) &= v10;
      goto LABEL_12;
    }
  }
  else
  {
    v10 = 0;
  }
  *(_QWORD *)(a2 + 16) = v9 & v10;
LABEL_12:
  if ( v4 )
  {
    v11 = (unsigned int)(v4 - 1);
    v12 = 8;
    v39 = 16 * v11 + 24;
    while ( 1 )
    {
      v28 = *(_BYTE *)(a1 - 16);
      if ( (v28 & 2) != 0 )
        v13 = *(_QWORD *)(a1 - 32);
      else
        v13 = -16 - 8LL * ((v28 >> 2) & 0xF) + a1;
      v29 = *(_QWORD *)(*(_QWORD *)(v13 + v12 - 8) + 136LL);
      v14 = *(_QWORD *)(*(_QWORD *)(v13 + v12) + 136LL);
      v51 = *(_DWORD *)(v14 + 32);
      if ( v51 > 0x40 )
      {
        v38 = v29;
        sub_C43780(&v50, v14 + 24);
        v29 = v38;
      }
      else
      {
        v50 = *(_QWORD *)(v14 + 24);
      }
      v49 = *(_DWORD *)(v29 + 32);
      if ( v49 > 0x40 )
        sub_C43780(&v48, v29 + 24);
      else
        v48 = *(_QWORD *)(v29 + 24);
      sub_AADC30(&v52, &v48, &v50);
      if ( v49 > 0x40 && v48 )
        j_j___libc_free_0_0(v48);
      if ( v51 > 0x40 && v50 )
        j_j___libc_free_0_0(v50);
      sub_AB0A00(&v48, &v52);
      sub_AB0910(&v46, &v52);
      v15 = v49;
      if ( v49 <= 0x40 )
        break;
      sub_C43C10(&v48, &v46);
      v15 = v49;
      v16 = v48;
      v49 = 0;
      v51 = v15;
      v50 = v48;
      if ( v15 <= 0x40 )
        goto LABEL_27;
      v35 = v48;
      v15 = sub_C444A0(&v50);
      if ( v35 )
      {
        v32 = v35;
        v36 = v15;
        j_j___libc_free_0_0(v32);
        v15 = v36;
      }
LABEL_29:
      if ( v47 > 0x40 && v46 )
      {
        v33 = v15;
        j_j___libc_free_0_0(v46);
        v15 = v33;
      }
      if ( v49 > 0x40 && v48 )
      {
        v34 = v15;
        j_j___libc_free_0_0(v48);
        v15 = v34;
      }
      v19 = v41;
      v43 = v41;
      if ( v41 > 0x40 )
      {
        v37 = v15;
        sub_C43690(&v42, 0, 0);
        v19 = v43;
        v15 = v37;
      }
      else
      {
        v42 = 0;
      }
      v20 = (unsigned int)v19 - v15;
      if ( (_DWORD)v19 != (_DWORD)v20 )
      {
        if ( (unsigned int)v20 > 0x3F || (unsigned int)v19 > 0x40 )
          sub_C43C90(&v42, v20, v19);
        else
          v42 |= 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v15) << ((unsigned __int8)v19 - (unsigned __int8)v15);
      }
      sub_AB0910(&v50, &v52);
      sub_C44AB0(&v44, &v50, v41);
      if ( v51 > 0x40 && v50 )
        j_j___libc_free_0_0(v50);
      v21 = v45;
      v49 = v45;
      if ( v45 <= 0x40 )
      {
        v22 = v44;
LABEL_46:
        v23 = v42 & v22;
        v48 = v23;
        goto LABEL_47;
      }
      sub_C43780(&v48, &v44);
      v21 = v49;
      if ( v49 <= 0x40 )
      {
        v22 = v48;
        goto LABEL_46;
      }
      sub_C43B90(&v48, &v42);
      v21 = v49;
      v23 = v48;
LABEL_47:
      v24 = *(_DWORD *)(a2 + 24) <= 0x40u;
      v51 = v21;
      v50 = v23;
      v49 = 0;
      if ( v24 )
      {
        *(_QWORD *)(a2 + 16) &= v23;
      }
      else
      {
        sub_C43B90(a2 + 16, &v50);
        v21 = v51;
      }
      if ( v21 > 0x40 && v50 )
        j_j___libc_free_0_0(v50);
      if ( v49 > 0x40 && v48 )
        j_j___libc_free_0_0(v48);
      v25 = v45;
      v47 = v45;
      if ( v45 <= 0x40 )
      {
        v26 = v44;
LABEL_57:
        v47 = 0;
        v27 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v25) & ~v26;
        if ( !v25 )
          v27 = 0;
        v46 = v27;
        goto LABEL_60;
      }
      sub_C43780(&v46, &v44);
      v25 = v47;
      if ( v47 <= 0x40 )
      {
        v26 = v46;
        goto LABEL_57;
      }
      sub_C43D10(&v46, &v44, v47, v30, v31);
      v25 = v47;
      v27 = v46;
      v47 = 0;
      v49 = v25;
      v48 = v46;
      if ( v25 > 0x40 )
      {
        sub_C43B90(&v48, &v42);
        v25 = v49;
        result = v48;
        goto LABEL_61;
      }
LABEL_60:
      result = v42 & v27;
      v48 = result;
LABEL_61:
      v24 = *(_DWORD *)(a2 + 8) <= 0x40u;
      v51 = v25;
      v50 = result;
      v49 = 0;
      if ( v24 )
      {
        *(_QWORD *)a2 &= result;
      }
      else
      {
        result = sub_C43B90(a2, &v50);
        v25 = v51;
      }
      if ( v25 > 0x40 && v50 )
        result = j_j___libc_free_0_0(v50);
      if ( v49 > 0x40 && v48 )
        result = j_j___libc_free_0_0(v48);
      if ( v47 > 0x40 && v46 )
        result = j_j___libc_free_0_0(v46);
      if ( v45 > 0x40 && v44 )
        result = j_j___libc_free_0_0(v44);
      if ( v43 > 0x40 && v42 )
        result = j_j___libc_free_0_0(v42);
      if ( v55 > 0x40 && v54 )
        result = j_j___libc_free_0_0(v54);
      if ( v53 > 0x40 )
      {
        if ( v52 )
          result = j_j___libc_free_0_0(v52);
      }
      v12 += 16;
      if ( v39 == v12 )
        return result;
    }
    v16 = v46 ^ v48;
    v49 = 0;
    v48 ^= v46;
LABEL_27:
    v17 = v15 - 64;
    if ( v16 )
    {
      _BitScanReverse64(&v18, v16);
      v15 = v17 + (v18 ^ 0x3F);
    }
    goto LABEL_29;
  }
  return result;
}
