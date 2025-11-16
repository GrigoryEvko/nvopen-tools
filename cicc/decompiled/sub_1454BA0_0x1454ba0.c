// Function: sub_1454BA0
// Address: 0x1454ba0
//
__int64 __fastcall sub_1454BA0(__int64 a1, unsigned int a2, __int64 a3)
{
  int v3; // r13d
  __int64 result; // rax
  __int64 v8; // rdx
  unsigned int v9; // ecx
  char v10; // r15
  __int64 v11; // r12
  bool v12; // cc
  unsigned __int8 *v13; // rdi
  unsigned __int8 *v14; // rsi
  unsigned __int8 *v15; // rdi
  unsigned __int8 *v16; // r9
  __int64 *v17; // r8
  __int64 v18; // rsi
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  __int64 v21; // rdi
  __int64 v22; // rax
  unsigned __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned __int8 *v26; // [rsp+0h] [rbp-60h]
  __int64 *v27; // [rsp+8h] [rbp-58h]
  __int64 *v28; // [rsp+8h] [rbp-58h]
  unsigned __int64 v29; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v30; // [rsp+18h] [rbp-48h]
  __int64 v31[8]; // [rsp+20h] [rbp-40h] BYREF

  *(_QWORD *)a1 = 0;
  *(_DWORD *)(a1 + 16) = 1;
  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 32) = 1;
  *(_QWORD *)(a1 + 24) = 0;
  v30 = a2;
  if ( a2 > 0x40 )
    sub_16A4EF0(&v29, 0, 0);
  else
    v29 = 0;
  result = *(unsigned __int16 *)(a3 + 24);
  if ( (_WORD)result == 4 )
  {
    if ( *(_QWORD *)(a3 + 40) != 2 || (result = *(_QWORD *)(a3 + 32), *(_WORD *)(*(_QWORD *)result + 24LL)) )
    {
      if ( v30 <= 0x40 )
        return result;
LABEL_6:
      if ( v29 )
        return j_j___libc_free_0_0(v29);
      return result;
    }
    v8 = *(_QWORD *)(*(_QWORD *)result + 32LL);
    if ( v30 <= 0x40 && (v9 = *(_DWORD *)(v8 + 32), v9 <= 0x40) )
    {
      v30 = *(_DWORD *)(v8 + 32);
      v29 = *(_QWORD *)(v8 + 24) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v9);
    }
    else
    {
      sub_16A51C0(&v29, v8 + 24);
      result = *(_QWORD *)(a3 + 32);
    }
    a3 = *(_QWORD *)(result + 8);
    result = *(unsigned __int16 *)(a3 + 24);
  }
  v10 = 0;
  if ( (unsigned __int16)(result - 1) <= 2u )
  {
    a3 = *(_QWORD *)(a3 + 32);
    v3 = (unsigned __int16)result;
    v10 = 1;
    result = *(unsigned __int16 *)(a3 + 24);
  }
  if ( (_WORD)result != 10 )
    goto LABEL_18;
  v11 = *(_QWORD *)(a3 - 8);
  if ( *(_BYTE *)(v11 + 16) != 79 )
    goto LABEL_18;
  result = *(_QWORD *)(v11 - 72);
  if ( !result )
    goto LABEL_18;
  *(_QWORD *)a1 = result;
  v13 = *(unsigned __int8 **)(v11 - 48);
  result = v13[16];
  v14 = v13 + 24;
  if ( (_BYTE)result == 13 )
    goto LABEL_23;
  if ( *(_BYTE *)(*(_QWORD *)v13 + 8LL) != 16
    || (unsigned __int8)result > 0x10u
    || (result = sub_15A1020(v13)) == 0
    || *(_BYTE *)(result + 16) != 13 )
  {
LABEL_18:
    v12 = v30 <= 0x40;
    *(_QWORD *)a1 = 0;
    if ( v12 )
      return result;
    goto LABEL_6;
  }
  v14 = (unsigned __int8 *)(result + 24);
LABEL_23:
  v15 = *(unsigned __int8 **)(v11 - 24);
  result = v15[16];
  v16 = v15 + 24;
  if ( (_BYTE)result == 13 )
    goto LABEL_24;
  if ( *(_BYTE *)(*(_QWORD *)v15 + 8LL) != 16 )
    goto LABEL_18;
  if ( (unsigned __int8)result > 0x10u )
    goto LABEL_18;
  result = sub_15A1020(v15);
  if ( !result || *(_BYTE *)(result + 16) != 13 )
    goto LABEL_18;
  v16 = (unsigned __int8 *)(result + 24);
LABEL_24:
  v17 = (__int64 *)(a1 + 8);
  if ( *(_DWORD *)(a1 + 16) <= 0x40u && *((_DWORD *)v14 + 2) <= 0x40u )
  {
    v21 = *(_QWORD *)v14;
    *(_QWORD *)(a1 + 8) = *(_QWORD *)v14;
    v22 = *((unsigned int *)v14 + 2);
    *(_DWORD *)(a1 + 16) = v22;
    v23 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v22;
    if ( (unsigned int)v22 > 0x40 )
    {
      v24 = (unsigned int)((unsigned __int64)(v22 + 63) >> 6) - 1;
      *(_QWORD *)(v21 + 8 * v24) &= v23;
    }
    else
    {
      *(_QWORD *)(a1 + 8) = v21 & v23;
    }
  }
  else
  {
    v26 = v16;
    sub_16A51C0(a1 + 8, v14);
    v16 = v26;
    v17 = (__int64 *)(a1 + 8);
  }
  if ( *(_DWORD *)(a1 + 32) <= 0x40u && *((_DWORD *)v16 + 2) <= 0x40u )
  {
    v18 = *(_QWORD *)v16;
    *(_QWORD *)(a1 + 24) = *(_QWORD *)v16;
    v19 = *((unsigned int *)v16 + 2);
    *(_DWORD *)(a1 + 32) = v19;
    v20 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v19;
    if ( (unsigned int)v19 > 0x40 )
    {
      v25 = (unsigned int)((unsigned __int64)(v19 + 63) >> 6) - 1;
      *(_QWORD *)(v18 + 8 * v25) &= v20;
    }
    else
    {
      *(_QWORD *)(a1 + 24) = v18 & v20;
    }
  }
  else
  {
    v27 = v17;
    sub_16A51C0(a1 + 24, v16);
    v17 = v27;
  }
  if ( v10 )
  {
    if ( v3 == 2 )
    {
      v28 = v17;
      sub_16A5C50(v31, v17, a2);
      sub_14536D0(v28, v31);
      sub_135E100(v31);
      sub_16A5C50(v31, a1 + 24, a2);
    }
    else
    {
      v28 = v17;
      if ( v3 == 3 )
      {
        sub_16A5B10(v31, v17, a2);
        sub_14536D0(v28, v31);
        sub_135E100(v31);
        sub_16A5B10(v31, a1 + 24, a2);
      }
      else
      {
        sub_16A5A50(v31, v17);
        sub_14536D0(v28, v31);
        sub_135E100(v31);
        sub_16A5A50(v31, a1 + 24);
      }
    }
    sub_14536D0((__int64 *)(a1 + 24), v31);
    sub_135E100(v31);
    v17 = v28;
  }
  sub_16A7200(v17, &v29);
  sub_16A7200(a1 + 24, &v29);
  return sub_135E100((__int64 *)&v29);
}
