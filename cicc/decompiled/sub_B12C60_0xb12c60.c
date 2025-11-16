// Function: sub_B12C60
// Address: 0xb12c60
//
__int64 __fastcall sub_B12C60(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 *v8; // rsi
  _BYTE *v9; // rsi
  __int64 v10; // rdx
  __int64 *v11; // rcx
  __int64 v12; // r8
  _QWORD *v13; // rbx
  unsigned __int8 *v14; // rax
  unsigned __int64 v15; // r10
  __int64 v16; // r9
  _QWORD *v17; // r8
  _QWORD *v18; // r13
  _QWORD *v19; // rax
  __int64 v20; // r14
  __int64 v21; // rdi
  _QWORD *v22; // rbx
  _QWORD *v23; // r13
  unsigned __int8 *v24; // rax
  __int64 v25; // rdx
  unsigned __int64 v26; // r9
  _BYTE *v27; // rdi
  __int64 v28; // r13
  __int64 *v29; // r15
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 *v32; // rax
  __int64 v33; // rbx
  __int64 result; // rax
  unsigned __int8 *v35; // [rsp+0h] [rbp-90h]
  unsigned __int8 *v38; // [rsp+10h] [rbp-80h]
  __int64 v39; // [rsp+20h] [rbp-70h] BYREF
  _QWORD *v40; // [rsp+28h] [rbp-68h]
  __int64 *v41; // [rsp+30h] [rbp-60h] BYREF
  __int64 v42; // [rsp+38h] [rbp-58h]
  _BYTE v43[80]; // [rsp+40h] [rbp-50h] BYREF

  sub_B11F20(&v41, a4);
  if ( *(_QWORD *)(a1 + 80) )
    sub_B91220(a1 + 80);
  v8 = v41;
  *(_QWORD *)(a1 + 80) = v41;
  if ( v8 )
    sub_B976B0(&v41, v8, a1 + 80, v5, v6, v7);
  v9 = (_BYTE *)a1;
  v41 = (__int64 *)v43;
  v42 = 0x400000000LL;
  sub_B129C0(&v39, a1);
  v12 = v39;
  v13 = v40;
  if ( v40 != (_QWORD *)v39 )
  {
    while ( 1 )
    {
      v16 = v12;
      v17 = (_QWORD *)(v12 & 0xFFFFFFFFFFFFFFF8LL);
      v16 >>= 2;
      v18 = v17;
      v19 = v17;
      v20 = v16 & 1;
      if ( (v16 & 1) != 0 )
        v19 = (_QWORD *)*v17;
      v21 = v19[17];
      if ( *(_BYTE *)v21 == 24 )
      {
        v14 = *(unsigned __int8 **)(v21 + 24);
        v9 = 0;
        v10 = (unsigned int)v42;
        if ( (unsigned int)*v14 - 1 > 1 )
          v14 = 0;
        v15 = (unsigned int)v42 + 1LL;
        if ( v15 <= HIDWORD(v42) )
          goto LABEL_10;
      }
      else
      {
        v14 = (unsigned __int8 *)sub_B98A20(v21, v9, v10, v11);
        v10 = (unsigned int)v42;
        v15 = (unsigned int)v42 + 1LL;
        if ( v15 <= HIDWORD(v42) )
          goto LABEL_10;
      }
      v9 = v43;
      v35 = v14;
      sub_C8D5F0(&v41, v43, v15, 8);
      v10 = (unsigned int)v42;
      v14 = v35;
LABEL_10:
      v11 = v41;
      v41[v10] = (__int64)v14;
      LODWORD(v42) = v42 + 1;
      if ( v20 || !v18 )
      {
        v12 = (unsigned __int64)(v18 + 1) | 4;
        if ( v13 == (_QWORD *)v12 )
          break;
      }
      else
      {
        v12 = (__int64)(v18 + 18);
        if ( v13 == v18 + 18 )
          break;
      }
    }
  }
  v22 = &a2[a3];
  if ( a2 != v22 )
  {
    v23 = a2;
    while ( 1 )
    {
      v27 = (_BYTE *)*v23;
      if ( *(_BYTE *)*v23 == 24 )
        break;
      v24 = (unsigned __int8 *)sub_B98A20(v27, v9, v10, v11);
      v25 = (unsigned int)v42;
      v26 = (unsigned int)v42 + 1LL;
      if ( v26 > HIDWORD(v42) )
        goto LABEL_27;
LABEL_24:
      v11 = v41;
      ++v23;
      v41[v25] = (__int64)v24;
      v10 = (unsigned int)(v42 + 1);
      LODWORD(v42) = v42 + 1;
      if ( v22 == v23 )
        goto LABEL_29;
    }
    v24 = (unsigned __int8 *)*((_QWORD *)v27 + 3);
    v25 = (unsigned int)v42;
    if ( (unsigned int)*v24 - 1 > 1 )
      v24 = 0;
    v26 = (unsigned int)v42 + 1LL;
    if ( v26 <= HIDWORD(v42) )
      goto LABEL_24;
LABEL_27:
    v9 = v43;
    v38 = v24;
    sub_C8D5F0(&v41, v43, v26, 8);
    v25 = (unsigned int)v42;
    v24 = v38;
    goto LABEL_24;
  }
  LODWORD(v10) = v42;
LABEL_29:
  v28 = (unsigned int)v10;
  v29 = v41;
  v30 = sub_B12A50(a1, 0);
  v32 = (__int64 *)sub_BD5C60(v30, 0, v31);
  v33 = sub_B00B60(v32, v29, v28);
  sub_B91340(a1 + 40, 0);
  *(_QWORD *)(a1 + 40) = v33;
  result = sub_B96F50(a1 + 40, 0);
  if ( v41 != (__int64 *)v43 )
    return _libc_free(v41, 0);
  return result;
}
