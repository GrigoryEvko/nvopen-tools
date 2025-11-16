// Function: sub_1112670
// Address: 0x1112670
//
__int64 __fastcall sub_1112670(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int8 *v3; // r12
  __int64 v4; // rax
  __int64 *v5; // rax
  __int64 v6; // rdx
  unsigned int v7; // r13d
  unsigned __int64 v8; // r14
  unsigned __int64 v9; // r14
  unsigned __int8 *v10; // rax
  unsigned int v11; // r15d
  unsigned __int8 *v12; // r13
  unsigned __int64 v13; // rax
  __int64 v14; // rax
  _BYTE **v15; // rax
  _BYTE *v16; // rsi
  __int64 result; // rax
  unsigned int v18; // r13d
  __int64 v19; // rdx
  unsigned __int64 v20; // r14
  unsigned __int8 *v21; // rsi
  unsigned int v22; // eax
  unsigned __int64 v23; // rdx
  __int64 v24; // r12
  unsigned __int64 v25; // rdx
  unsigned int v26; // ebx
  unsigned __int64 *v27; // r12
  __int64 v28; // rdx
  int v29; // r14d
  unsigned __int8 *v30; // rax
  unsigned int v31; // r15d
  unsigned __int8 *v32; // r13
  __int64 v33; // rdx
  unsigned __int64 v34; // r14
  __int64 v35; // [rsp-8h] [rbp-78h]
  __int64 v36; // [rsp-8h] [rbp-78h]
  unsigned __int64 *v37; // [rsp+0h] [rbp-70h] BYREF
  unsigned int v38; // [rsp+8h] [rbp-68h]
  unsigned __int64 *v39; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v40; // [rsp+18h] [rbp-58h]
  __int64 v41; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v42; // [rsp+28h] [rbp-48h]
  __int64 v43; // [rsp+30h] [rbp-40h]
  unsigned int v44; // [rsp+38h] [rbp-38h]

  v3 = *(unsigned __int8 **)a1;
  if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 8LL) - 17 > 1
    || (a2 = 0, (v3 = sub_AD7630(*(_QWORD *)a1, 0, a3)) != 0) )
  {
    if ( !sub_AC30F0((__int64)v3) )
    {
      v29 = *(_DWORD *)(a1 + 8);
      v30 = sub_AD8340(v3, a2, v28);
      v31 = *((_DWORD *)v30 + 2);
      v32 = v30;
      if ( v31 <= 0x40 )
      {
        v33 = *(_QWORD *)v30;
      }
      else
      {
        if ( v31 - (unsigned int)sub_C444A0((__int64)v30) > 0x40 )
          goto LABEL_3;
        v33 = **(_QWORD **)v32;
      }
      if ( v29 - 1 != v33 )
      {
        v4 = *(_QWORD *)(a1 + 16);
        if ( (*(_BYTE *)(v4 + 7) & 0x40) == 0 )
          goto LABEL_4;
        goto LABEL_44;
      }
    }
    return 1;
  }
LABEL_3:
  v4 = *(_QWORD *)(a1 + 16);
  if ( (*(_BYTE *)(v4 + 7) & 0x40) == 0 )
  {
LABEL_4:
    v5 = (__int64 *)(v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF));
    goto LABEL_5;
  }
LABEL_44:
  v5 = *(__int64 **)(v4 - 8);
LABEL_5:
  if ( *(_BYTE *)*v5 > 0x15u )
    goto LABEL_20;
  sub_9AC3E0((__int64)&v41, *v5, *(_QWORD *)(a1 + 24), 0, 0, 0, 0, 1);
  v7 = v42;
  if ( v42 > 0x40 )
  {
    v8 = (unsigned int)sub_C44500((__int64)&v41);
  }
  else
  {
    if ( !v42 )
      goto LABEL_54;
    v8 = 64;
    if ( v41 << (64 - (unsigned __int8)v42) != -1 )
    {
      _BitScanReverse64(&v9, ~(v41 << (64 - (unsigned __int8)v42)));
      v8 = (unsigned int)v9 ^ 0x3F;
    }
  }
  if ( v7 - (unsigned int)v8 <= 1 )
    goto LABEL_54;
  if ( v3 )
  {
    v10 = sub_AD8340(v3, v35, v6);
    v11 = *((_DWORD *)v10 + 2);
    v12 = v10;
    if ( v11 <= 0x40 )
    {
      v13 = *(_QWORD *)v10;
LABEL_15:
      if ( v8 >= v13 )
        goto LABEL_54;
      goto LABEL_16;
    }
    if ( v11 - (unsigned int)sub_C444A0((__int64)v10) <= 0x40 )
    {
      v13 = **(_QWORD **)v12;
      goto LABEL_15;
    }
  }
LABEL_16:
  if ( v44 > 0x40 && v43 )
    j_j___libc_free_0_0(v43);
  if ( v42 > 0x40 && v41 )
    j_j___libc_free_0_0(v41);
LABEL_20:
  v14 = *(_QWORD *)(a1 + 96);
  if ( (*(_BYTE *)(v14 + 7) & 0x40) != 0 )
    v15 = *(_BYTE ***)(v14 - 8);
  else
    v15 = (_BYTE **)(v14 - 32LL * (*(_DWORD *)(v14 + 4) & 0x7FFFFFF));
  v16 = *v15;
  result = 0;
  if ( *v16 > 0x15u )
    return result;
  sub_9AC3E0((__int64)&v41, (__int64)v16, *(_QWORD *)(a1 + 24), 0, 0, 0, 0, 1);
  v18 = v42;
  v19 = v36;
  if ( v42 > 0x40 )
  {
    v20 = (unsigned int)sub_C44500((__int64)&v41);
    goto LABEL_25;
  }
  if ( !v42 )
  {
LABEL_54:
    if ( v44 > 0x40 && v43 )
      j_j___libc_free_0_0(v43);
    if ( v42 > 0x40 && v41 )
      j_j___libc_free_0_0(v41);
    return 1;
  }
  v20 = 64;
  if ( v41 << (64 - (unsigned __int8)v42) != -1 )
  {
    _BitScanReverse64(&v34, ~(v41 << (64 - (unsigned __int8)v42)));
    v20 = (unsigned int)v34 ^ 0x3F;
  }
LABEL_25:
  if ( v18 - (unsigned int)v20 <= 1 )
    goto LABEL_54;
  if ( !v3 )
    goto LABEL_34;
  v21 = sub_AD8340(v3, (__int64)v16, v19);
  v22 = *((_DWORD *)v21 + 2);
  v40 = v22;
  if ( v22 > 0x40 )
  {
    sub_C43780((__int64)&v39, (const void **)v21);
    v24 = (unsigned int)(*(_DWORD *)(a1 + 8) - 1);
    v22 = v40;
    if ( v40 > 0x40 )
    {
      sub_C43D10((__int64)&v39);
      goto LABEL_32;
    }
    v23 = (unsigned __int64)v39;
  }
  else
  {
    v23 = *(_QWORD *)v21;
    v24 = (unsigned int)(*(_DWORD *)(a1 + 8) - 1);
  }
  v25 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v22) & ~v23;
  if ( !v22 )
    v25 = 0;
  v39 = (unsigned __int64 *)v25;
LABEL_32:
  sub_C46250((__int64)&v39);
  sub_C46A40((__int64)&v39, v24);
  v26 = v40;
  v27 = v39;
  v38 = v40;
  v37 = v39;
  if ( v40 <= 0x40 )
  {
    if ( v20 < (unsigned __int64)v39 )
      goto LABEL_34;
    goto LABEL_54;
  }
  if ( v26 - (unsigned int)sub_C444A0((__int64)&v37) > 0x40 )
  {
    if ( !v27 )
      goto LABEL_34;
  }
  else if ( v20 >= *v27 )
  {
    j_j___libc_free_0_0(v27);
    goto LABEL_54;
  }
  j_j___libc_free_0_0(v27);
LABEL_34:
  if ( v44 > 0x40 && v43 )
    j_j___libc_free_0_0(v43);
  if ( v42 > 0x40 )
  {
    if ( v41 )
      j_j___libc_free_0_0(v41);
  }
  return 0;
}
