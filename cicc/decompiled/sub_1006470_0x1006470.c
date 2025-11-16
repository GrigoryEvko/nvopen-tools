// Function: sub_1006470
// Address: 0x1006470
//
__int64 __fastcall sub_1006470(unsigned __int8 *a1, __int64 a2, __m128i *a3)
{
  __int64 result; // rax
  unsigned __int64 v5; // rdx
  __int64 v6; // rcx
  int v7; // eax
  __int64 v8; // rbx
  unsigned __int8 *v9; // rdx
  unsigned __int64 v10; // rax
  unsigned __int8 *v11; // rax
  unsigned __int64 v12; // rdx
  __int64 v13; // rsi
  int v14; // edx
  __int64 v15; // rdx
  unsigned int v16; // r12d
  unsigned int v17; // edx
  unsigned __int64 v18; // rax
  unsigned int v19; // r14d
  unsigned __int64 v20; // r13
  _QWORD *v21; // rax
  __int64 v22; // rdx
  _BYTE *v23; // rax
  __int64 v24; // rsi
  int v25; // eax
  bool v26; // zf
  __int64 v27; // [rsp-70h] [rbp-70h]
  __int64 v28; // [rsp-70h] [rbp-70h]
  __int64 v29; // [rsp-60h] [rbp-60h] BYREF
  __int64 v30; // [rsp-58h] [rbp-58h] BYREF
  const void **v31; // [rsp-50h] [rbp-50h] BYREF
  __int64 *v32; // [rsp-48h] [rbp-48h] BYREF
  const void ***v33; // [rsp-40h] [rbp-40h] BYREF
  __int64 v34; // [rsp-38h] [rbp-38h]
  __int64 *v35; // [rsp-30h] [rbp-30h]

  if ( !a3[4].m128i_i8[0] )
    return 0;
  v5 = *a1;
  if ( (unsigned __int8)v5 <= 0x1Cu )
  {
    if ( (_BYTE)v5 != 5 )
      goto LABEL_12;
    v7 = *((unsigned __int16 *)a1 + 1);
    if ( (*((_WORD *)a1 + 1) & 0xFFF7) != 0x11 && (v7 & 0xFFFD) != 0xD )
      goto LABEL_12;
  }
  else
  {
    if ( (unsigned __int8)v5 > 0x36u )
      goto LABEL_12;
    v6 = 0x40540000000000LL;
    v7 = (unsigned __int8)v5 - 29;
    if ( !_bittest64(&v6, v5) )
      goto LABEL_12;
  }
  if ( v7 == 25 && (a1[1] & 2) != 0 )
  {
    result = *((_QWORD *)a1 - 8);
    if ( result )
    {
      v29 = *((_QWORD *)a1 - 8);
      if ( a2 == *((_QWORD *)a1 - 4) )
        return result;
    }
  }
LABEL_12:
  v8 = a2 + 24;
  if ( *(_BYTE *)a2 != 17 )
  {
    v22 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a2 + 8) + 8LL) - 17;
    if ( (unsigned int)v22 > 1 )
      return 0;
    if ( *(_BYTE *)a2 > 0x15u )
      return 0;
    v23 = sub_AD7630(a2, 0, v22);
    if ( !v23 || *v23 != 17 )
      return 0;
    LOBYTE(v5) = *a1;
    v8 = (__int64)(v23 + 24);
  }
  LOBYTE(v34) = 0;
  v32 = &v29;
  v33 = &v31;
  v35 = &v30;
  if ( (_BYTE)v5 != 58 )
    return 0;
  v9 = (unsigned __int8 *)*((_QWORD *)a1 - 8);
  v10 = *v9;
  if ( (unsigned __int8)v10 > 0x1Cu )
  {
    if ( (unsigned __int8)v10 > 0x36u )
      goto LABEL_18;
    v24 = 0x40540000000000LL;
    if ( !_bittest64(&v24, v10) )
      goto LABEL_18;
    v25 = (unsigned __int8)v10 - 29;
  }
  else
  {
    if ( (_BYTE)v10 != 5 )
      goto LABEL_18;
    v25 = *((unsigned __int16 *)v9 + 1);
    if ( (*((_WORD *)v9 + 1) & 0xFFFD) != 0xD && (v25 & 0xFFF7) != 0x11 )
      goto LABEL_18;
  }
  if ( v25 == 25 && (v9[1] & 2) != 0 && *((_QWORD *)v9 - 8) )
  {
    v29 = *((_QWORD *)v9 - 8);
    v26 = (unsigned __int8)sub_991580((__int64)&v33, *((_QWORD *)v9 - 4)) == 0;
    v11 = (unsigned __int8 *)*((_QWORD *)a1 - 4);
    if ( !v26 && v11 )
      goto LABEL_28;
    goto LABEL_19;
  }
LABEL_18:
  v11 = (unsigned __int8 *)*((_QWORD *)a1 - 4);
LABEL_19:
  v12 = *v11;
  if ( (unsigned __int8)v12 <= 0x1Cu )
  {
    if ( (_BYTE)v12 != 5 )
      return 0;
    v14 = *((unsigned __int16 *)v11 + 1);
    if ( (*((_WORD *)v11 + 1) & 0xFFFD) != 0xD && (v14 & 0xFFF7) != 0x11 )
      return 0;
  }
  else
  {
    if ( (unsigned __int8)v12 > 0x36u )
      return 0;
    v13 = 0x40540000000000LL;
    if ( !_bittest64(&v13, v12) )
      return 0;
    v14 = (unsigned __int8)v12 - 29;
  }
  if ( v14 != 25 )
    return 0;
  if ( (v11[1] & 2) == 0 )
    return 0;
  v15 = *((_QWORD *)v11 - 8);
  if ( !v15 )
    return 0;
  *v32 = v15;
  if ( !(unsigned __int8)sub_991580((__int64)&v33, *((_QWORD *)v11 - 4)) )
    return 0;
  v11 = (unsigned __int8 *)*((_QWORD *)a1 - 8);
  if ( !v11 )
    return 0;
LABEL_28:
  *v35 = (__int64)v11;
  if ( *(_DWORD *)(v8 + 8) <= 0x40u )
  {
    if ( *(const void **)v8 != *v31 )
      return 0;
  }
  else if ( !sub_C43C50(v8, v31) )
  {
    return 0;
  }
  sub_9AC330((__int64)&v32, v30, 0, a3);
  v16 = (unsigned int)v33;
  if ( (unsigned int)v33 > 0x40 )
  {
    v17 = v16 - sub_C44500((__int64)&v32);
  }
  else if ( (_DWORD)v33 )
  {
    v17 = (_DWORD)v33 - 64;
    if ( (_QWORD)v32 << (64 - (unsigned __int8)v33) != -1 )
    {
      _BitScanReverse64(&v18, ~((_QWORD)v32 << (64 - (unsigned __int8)v33)));
      v17 = (_DWORD)v33 - (v18 ^ 0x3F);
    }
  }
  else
  {
    v17 = 0;
  }
  v19 = *(_DWORD *)(v8 + 8);
  v20 = v17;
  if ( v19 > 0x40 )
  {
    if ( v19 - (unsigned int)sub_C444A0(v8) > 0x40 )
      goto LABEL_62;
    v21 = **(_QWORD ***)v8;
  }
  else
  {
    v21 = *(_QWORD **)v8;
  }
  if ( v20 > (unsigned __int64)v21 )
  {
    if ( (unsigned int)v35 > 0x40 && v34 )
    {
      j_j___libc_free_0_0(v34);
      v16 = (unsigned int)v33;
    }
    if ( v16 > 0x40 )
    {
      if ( v32 )
        j_j___libc_free_0_0(v32);
    }
    return 0;
  }
LABEL_62:
  result = v29;
  if ( (unsigned int)v35 > 0x40 && v34 )
  {
    v27 = v29;
    j_j___libc_free_0_0(v34);
    v16 = (unsigned int)v33;
    result = v27;
  }
  if ( v16 > 0x40 && v32 )
  {
    v28 = result;
    j_j___libc_free_0_0(v32);
    return v28;
  }
  return result;
}
