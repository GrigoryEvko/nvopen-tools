// Function: sub_B59230
// Address: 0xb59230
//
__int64 __fastcall sub_B59230(__int64 a1, _QWORD *a2, _BYTE *a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rcx
  _BYTE *v11; // rsi
  __int64 v12; // rdx
  __int64 *v13; // rcx
  __int64 v14; // rax
  _QWORD *v15; // r13
  __int64 v16; // r9
  _QWORD *v17; // rax
  _QWORD *v18; // rbx
  __int64 v19; // r14
  __int64 v20; // rdi
  unsigned __int8 *v21; // rax
  unsigned __int64 v22; // r10
  _BYTE *v23; // rsi
  _QWORD *v24; // r14
  _QWORD *v25; // rbx
  unsigned __int8 *v26; // r13
  __int64 v27; // rax
  unsigned __int64 v28; // rdx
  _BYTE *v29; // rdi
  __int64 v30; // rax
  __int64 *v31; // r14
  __int64 v32; // r13
  __int64 *v33; // rax
  __int64 v34; // r13
  __int64 v35; // rax
  __int64 v36; // rsi
  __int64 result; // rax
  __int64 v38; // r12
  __int64 v39; // rdx
  __int64 v40; // rdx
  unsigned __int8 *v41; // [rsp+8h] [rbp-88h]
  __int64 v44; // [rsp+20h] [rbp-70h] BYREF
  _QWORD *v45; // [rsp+28h] [rbp-68h]
  __int64 *v46; // [rsp+30h] [rbp-60h] BYREF
  __int64 v47; // [rsp+38h] [rbp-58h]
  _BYTE v48[80]; // [rsp+40h] [rbp-50h] BYREF

  v6 = sub_BD5C60(a1, a2);
  v7 = sub_B9F6F0(v6, a4);
  v8 = a1 + 32 * (2LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  if ( *(_QWORD *)v8 )
  {
    v9 = *(_QWORD *)(v8 + 8);
    **(_QWORD **)(v8 + 16) = v9;
    if ( v9 )
      *(_QWORD *)(v9 + 16) = *(_QWORD *)(v8 + 16);
  }
  *(_QWORD *)v8 = v7;
  if ( v7 )
  {
    v10 = *(_QWORD *)(v7 + 16);
    *(_QWORD *)(v8 + 8) = v10;
    if ( v10 )
      *(_QWORD *)(v10 + 16) = v8 + 8;
    *(_QWORD *)(v8 + 16) = v7 + 16;
    *(_QWORD *)(v7 + 16) = v8;
  }
  v11 = (_BYTE *)a1;
  v47 = 0x400000000LL;
  v46 = (__int64 *)v48;
  sub_B58E30(&v44, a1);
  v14 = v44;
  v15 = v45;
  if ( v45 != (_QWORD *)v44 )
  {
    while ( 1 )
    {
      v16 = v14;
      v17 = (_QWORD *)(v14 & 0xFFFFFFFFFFFFFFF8LL);
      v16 >>= 2;
      v18 = v17;
      v19 = v16 & 1;
      if ( (v16 & 1) != 0 )
        v17 = (_QWORD *)*v17;
      v20 = v17[17];
      if ( *(_BYTE *)v20 == 24 )
        break;
      v21 = (unsigned __int8 *)sub_B98A20(v20, v11, v12, v13);
      v12 = (unsigned int)v47;
      v22 = (unsigned int)v47 + 1LL;
      if ( v22 > HIDWORD(v47) )
        goto LABEL_29;
LABEL_17:
      v13 = v46;
      v46[v12] = (__int64)v21;
      LODWORD(v47) = v47 + 1;
      if ( v19 )
      {
        v14 = (unsigned __int64)(v18 + 1) | 4;
        if ( v15 == (_QWORD *)v14 )
          goto LABEL_19;
      }
      else
      {
        v14 = (__int64)(v18 + 18);
        if ( v15 == v18 + 18 )
          goto LABEL_19;
      }
    }
    v21 = *(unsigned __int8 **)(v20 + 24);
    v11 = 0;
    v12 = (unsigned int)v47;
    if ( (unsigned int)*v21 - 1 > 1 )
      v21 = 0;
    v22 = (unsigned int)v47 + 1LL;
    if ( v22 <= HIDWORD(v47) )
      goto LABEL_17;
LABEL_29:
    v11 = v48;
    v41 = v21;
    sub_C8D5F0(&v46, v48, v22, 8);
    v12 = (unsigned int)v47;
    v21 = v41;
    goto LABEL_17;
  }
LABEL_19:
  v23 = a3;
  v24 = &a2[(_QWORD)a3];
  if ( a2 != v24 )
  {
    v25 = a2;
    while ( 1 )
    {
      v29 = (_BYTE *)*v25;
      if ( *(_BYTE *)*v25 == 24 )
        break;
      v30 = sub_B98A20(v29, v23, v12, v13);
      v13 = (__int64 *)HIDWORD(v47);
      v26 = (unsigned __int8 *)v30;
      v27 = (unsigned int)v47;
      v28 = (unsigned int)v47 + 1LL;
      if ( v28 > HIDWORD(v47) )
        goto LABEL_27;
LABEL_24:
      ++v25;
      v46[v27] = (__int64)v26;
      v12 = (unsigned int)(v47 + 1);
      LODWORD(v47) = v47 + 1;
      if ( v24 == v25 )
        goto LABEL_31;
    }
    v26 = (unsigned __int8 *)*((_QWORD *)v29 + 3);
    v13 = (__int64 *)HIDWORD(v47);
    if ( (unsigned int)*v26 - 1 > 1 )
      v26 = 0;
    v27 = (unsigned int)v47;
    v28 = (unsigned int)v47 + 1LL;
    if ( v28 <= HIDWORD(v47) )
      goto LABEL_24;
LABEL_27:
    v23 = v48;
    sub_C8D5F0(&v46, v48, v28, 8);
    v27 = (unsigned int)v47;
    goto LABEL_24;
  }
  LODWORD(v12) = v47;
LABEL_31:
  v31 = v46;
  v32 = (unsigned int)v12;
  v33 = (__int64 *)sub_BD5C60(a1, v23);
  v34 = sub_B00B60(v33, v31, v32);
  v35 = sub_BD5C60(a1, v31);
  v36 = v34;
  result = sub_B9F6F0(v35, v34);
  v38 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  if ( *(_QWORD *)v38 )
  {
    v39 = *(_QWORD *)(v38 + 8);
    **(_QWORD **)(v38 + 16) = v39;
    if ( v39 )
      *(_QWORD *)(v39 + 16) = *(_QWORD *)(v38 + 16);
  }
  *(_QWORD *)v38 = result;
  if ( result )
  {
    v40 = *(_QWORD *)(result + 16);
    *(_QWORD *)(v38 + 8) = v40;
    if ( v40 )
    {
      v36 = v38 + 8;
      *(_QWORD *)(v40 + 16) = v38 + 8;
    }
    *(_QWORD *)(v38 + 16) = result + 16;
    *(_QWORD *)(result + 16) = v38;
  }
  if ( v46 != (__int64 *)v48 )
    return _libc_free(v46, v36);
  return result;
}
