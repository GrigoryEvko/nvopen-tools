// Function: sub_101CD30
// Address: 0x101cd30
//
unsigned __int8 *__fastcall sub_101CD30(unsigned int a1, _BYTE *a2, _BYTE *a3, char a4, __m128i *a5, int a6)
{
  _BYTE *v10; // r12
  _BYTE *v12; // rdi
  __int64 *v13; // r9
  __int64 *v14; // rsi
  char v15; // al
  unsigned __int64 v16; // rdx
  unsigned int v17; // eax
  unsigned int v18; // ebx
  unsigned int v19; // edx
  unsigned int v21; // ebx
  unsigned __int8 *v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdi
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rsi
  __int64 v29; // rsi
  unsigned __int64 v30; // [rsp+8h] [rbp-A8h]
  _BYTE *v31; // [rsp+10h] [rbp-A0h] BYREF
  _BYTE *v32; // [rsp+18h] [rbp-98h] BYREF
  __int64 v33; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v34; // [rsp+28h] [rbp-88h]
  unsigned __int64 v35; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v36; // [rsp+38h] [rbp-78h]
  __int64 v37; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v38; // [rsp+48h] [rbp-68h]
  __int64 v39; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v40; // [rsp+58h] [rbp-58h]
  unsigned __int64 v41; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v42; // [rsp+68h] [rbp-48h]
  __int64 v43; // [rsp+70h] [rbp-40h] BYREF
  unsigned int v44; // [rsp+78h] [rbp-38h]

  v32 = a2;
  v31 = a3;
  v10 = (_BYTE *)sub_FFE3E0(a1, &v32, &v31, a5->m128i_i64);
  if ( v10 )
    return v10;
  if ( *v32 == 13 )
    return v32;
  if ( (unsigned __int8)sub_FFFE90((__int64)v32) )
    return (unsigned __int8 *)sub_AD6530(*((_QWORD *)v32 + 1), (__int64)&v32);
  if ( (unsigned __int8)sub_FFFE90((__int64)v31) )
    return v32;
  v12 = v31;
  if ( *v31 == 69 )
  {
    v23 = *((_QWORD *)v31 - 4);
    if ( v23 )
    {
      v24 = *(_QWORD *)(v23 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v24 + 8) - 17 <= 1 )
        v24 = **(_QWORD **)(v24 + 16);
      if ( !sub_BCAC40(v24, 1) )
      {
        v12 = v31;
        goto LABEL_7;
      }
      return v32;
    }
  }
LABEL_7:
  if ( (unsigned __int8)sub_1003410((__int64)v12, (__int64)a5) )
    return (unsigned __int8 *)sub_ACADE0(*((__int64 ***)v32 + 1));
  v13 = (__int64 *)v32;
  v14 = (__int64 *)v31;
  v15 = *v32;
  if ( *v32 == 86 || *v31 == 86 )
  {
    v22 = sub_101C8A0(a1, (__int64 *)v32, (__int64 *)v31, a5, a6);
    if ( v22 )
      return v22;
    v13 = (__int64 *)v32;
    v14 = (__int64 *)v31;
    v15 = *v32;
  }
  if ( v15 == 84 || *(_BYTE *)v14 == 84 )
  {
    v22 = sub_101CAB0(a1, v13, v14, a5, a6);
    if ( !v22 )
    {
      v14 = (__int64 *)v31;
      goto LABEL_12;
    }
    return v22;
  }
LABEL_12:
  sub_9AC330((__int64)&v33, (__int64)v14, 0, a5);
  v42 = v36;
  if ( v36 > 0x40 )
  {
    sub_C43780((__int64)&v41, (const void **)&v35);
    v21 = v42;
    v16 = v34;
    if ( v42 > 0x40 )
    {
      v30 = v34;
      if ( v21 - (unsigned int)sub_C444A0((__int64)&v41) <= 0x40 && v30 > *(_QWORD *)v41 )
      {
        j_j___libc_free_0_0(v41);
        v17 = v34;
        goto LABEL_16;
      }
      if ( v41 )
        j_j___libc_free_0_0(v41);
LABEL_35:
      v10 = (_BYTE *)sub_ACADE0(*((__int64 ***)v32 + 1));
      goto LABEL_21;
    }
  }
  else
  {
    v16 = v34;
    v41 = v35;
  }
  if ( v16 <= v41 )
    goto LABEL_35;
  v17 = v34;
LABEL_16:
  v18 = v17 - 1;
  if ( v17 == 1 || (_BitScanReverse(&v19, v18), v18 = 32 - (v19 ^ 0x1F), v17 <= 0x40) )
  {
    _RAX = ~v33;
    if ( v33 == -1 )
      goto LABEL_49;
    __asm { tzcnt   rax, rax }
  }
  else
  {
    LODWORD(_RAX) = sub_C445E0((__int64)&v33);
  }
  if ( (unsigned int)_RAX >= v18 )
  {
LABEL_49:
    v10 = v32;
    goto LABEL_21;
  }
  if ( !a4 )
    goto LABEL_21;
  sub_9AC330((__int64)&v37, (__int64)v32, 0, a5);
  sub_C74E10((__int64)&v41, (__int64)&v37, (__int64)&v33, 0, 0, 0);
  if ( v38 > 0x40 )
    v26 = *(_QWORD *)(v37 + 8LL * ((v38 - 1) >> 6));
  else
    v26 = v37;
  if ( (v26 & (1LL << ((unsigned __int8)v38 - 1))) != 0 )
  {
    v28 = 1LL << ((unsigned __int8)v42 - 1);
    if ( v42 > 0x40 )
      *(_QWORD *)(v41 + 8LL * ((v42 - 1) >> 6)) |= v28;
    else
      v41 |= v28;
  }
  v27 = v39;
  if ( v40 > 0x40 )
    v27 = *(_QWORD *)(v39 + 8LL * ((v40 - 1) >> 6));
  if ( (v27 & (1LL << ((unsigned __int8)v40 - 1))) != 0 )
  {
    v29 = 1LL << ((unsigned __int8)v44 - 1);
    if ( v44 > 0x40 )
      *(_QWORD *)(v43 + 8LL * ((v44 - 1) >> 6)) |= v29;
    else
      v43 |= v29;
  }
  if ( v42 <= 0x40 )
  {
    if ( (v43 & v41) == 0 )
      goto LABEL_60;
    goto LABEL_59;
  }
  if ( (unsigned __int8)sub_C446A0((__int64 *)&v41, &v43) )
LABEL_59:
    v10 = (_BYTE *)sub_ACADE0(*((__int64 ***)v32 + 1));
LABEL_60:
  sub_969240(&v43);
  sub_969240((__int64 *)&v41);
  sub_969240(&v39);
  sub_969240(&v37);
LABEL_21:
  if ( v36 > 0x40 && v35 )
    j_j___libc_free_0_0(v35);
  if ( v34 > 0x40 && v33 )
    j_j___libc_free_0_0(v33);
  return v10;
}
