// Function: sub_B59720
// Address: 0xb59720
//
__int64 *__fastcall sub_B59720(__int64 a1, __int64 a2, unsigned __int8 *a3)
{
  __int64 v3; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 *v8; // rsi
  __int64 *result; // rax
  _QWORD *v10; // r14
  __int64 v11; // rdx
  _BYTE *v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // r12
  __int64 v16; // rax
  unsigned __int8 *v17; // r12
  __int64 v18; // rax
  __int64 v19; // r8
  _QWORD *v20; // rax
  _QWORD *v21; // rbx
  __int64 v22; // r8
  __int64 v23; // rdi
  _QWORD *v24; // rax
  unsigned __int8 *v25; // rax
  __int64 v26; // rdx
  unsigned __int64 v27; // r10
  __int64 *v28; // r14
  __int64 v29; // r12
  __int64 *v30; // rax
  __int64 v31; // r12
  __int64 v32; // rax
  __int64 v33; // rsi
  __int64 v34; // r15
  __int64 v35; // rdx
  __int64 v36; // rdx
  __int64 v37; // rax
  unsigned int v38; // edx
  unsigned __int8 *v39; // [rsp+8h] [rbp-98h]
  __int64 v40; // [rsp+10h] [rbp-90h]
  __int64 v41; // [rsp+10h] [rbp-90h]
  __int64 v42; // [rsp+18h] [rbp-88h] BYREF
  __int64 v43; // [rsp+28h] [rbp-78h] BYREF
  _QWORD *v44; // [rsp+30h] [rbp-70h] BYREF
  _QWORD *v45; // [rsp+38h] [rbp-68h]
  _BYTE *v46; // [rsp+40h] [rbp-60h] BYREF
  __int64 v47; // [rsp+48h] [rbp-58h]
  _BYTE v48[80]; // [rsp+50h] [rbp-50h] BYREF

  v3 = *(_QWORD *)(a1 - 32);
  v42 = a2;
  if ( !v3 || *(_BYTE *)v3 || *(_QWORD *)(v3 + 24) != *(_QWORD *)(a1 + 80) )
    BUG();
  if ( *(_DWORD *)(v3 + 36) == 68 && v42 == sub_B595C0(a1) )
    sub_B59690(a1, (__int64)a3, v6, v7);
  sub_B58E30(&v44, a1);
  v8 = (__int64 *)&v44;
  result = sub_B58EE0(&v43, (__int64 *)&v44, &v42);
  v10 = v45;
  if ( v45 == (_QWORD *)v43 )
    return result;
  result = (__int64 *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  v11 = *a3;
  v12 = *(_BYTE **)(*result + 24);
  if ( *v12 != 4 )
  {
    if ( (_BYTE)v11 == 24 )
    {
      v13 = result[1];
      *(_QWORD *)result[2] = v13;
      if ( !v13 )
      {
        *result = (__int64)a3;
LABEL_13:
        v14 = *((_QWORD *)a3 + 2);
        result[1] = v14;
        if ( v14 )
          *(_QWORD *)(v14 + 16) = result + 1;
        result[2] = (__int64)(a3 + 16);
        *((_QWORD *)a3 + 2) = result;
        return result;
      }
    }
    else
    {
      v15 = sub_B98A20(a3, &v44, v11, v12);
      v16 = sub_BD5C60(a1, &v44);
      a3 = (unsigned __int8 *)sub_B9F6F0(v16, v15);
      result = (__int64 *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
      if ( !*result || (v13 = result[1], (*(_QWORD *)result[2] = v13) == 0) )
      {
LABEL_12:
        *result = (__int64)a3;
        if ( !a3 )
          return result;
        goto LABEL_13;
      }
    }
    *(_QWORD *)(v13 + 16) = result[2];
    goto LABEL_12;
  }
  v46 = v48;
  v47 = 0x400000000LL;
  if ( (_BYTE)v11 == 24 )
  {
    v17 = (unsigned __int8 *)*((_QWORD *)a3 + 3);
    if ( (unsigned int)*v17 - 1 >= 2 )
      v17 = 0;
  }
  else
  {
    v37 = sub_B98A20(a3, &v44, v11, v12);
    v10 = v45;
    v17 = (unsigned __int8 *)v37;
  }
  v18 = (__int64)v44;
  if ( v10 != v44 )
  {
    while ( 1 )
    {
      v19 = v18;
      v20 = (_QWORD *)(v18 & 0xFFFFFFFFFFFFFFF8LL);
      v21 = v20;
      v22 = (v19 >> 2) & 1;
      if ( (_DWORD)v22 )
        v20 = (_QWORD *)*v20;
      v23 = v20[17];
      v24 = (_QWORD *)(v43 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v43 & 4) != 0 )
        v24 = (_QWORD *)*v24;
      if ( v24[17] != v23 )
        break;
      v26 = (unsigned int)v47;
      v25 = v17;
      v27 = (unsigned int)v47 + 1LL;
      if ( v27 > HIDWORD(v47) )
        goto LABEL_49;
LABEL_35:
      v12 = v46;
      *(_QWORD *)&v46[8 * v26] = v25;
      v38 = v47 + 1;
      LODWORD(v47) = v47 + 1;
      if ( v22 )
      {
        v18 = (unsigned __int64)(v21 + 1) | 4;
        if ( v10 == (_QWORD *)v18 )
          goto LABEL_37;
      }
      else
      {
        v18 = (__int64)(v21 + 18);
        if ( v10 == v21 + 18 )
          goto LABEL_37;
      }
    }
    if ( *(_BYTE *)v23 == 24 )
    {
      v25 = *(unsigned __int8 **)(v23 + 24);
      v8 = 0;
      if ( (unsigned int)*v25 - 1 > 1 )
        v25 = 0;
    }
    else
    {
      v41 = (unsigned int)v22;
      v25 = (unsigned __int8 *)sub_B98A20(v23, v8, v43 & 4, v12);
      v22 = v41;
    }
    v26 = (unsigned int)v47;
    v27 = (unsigned int)v47 + 1LL;
    if ( v27 <= HIDWORD(v47) )
      goto LABEL_35;
LABEL_49:
    v8 = (__int64 *)v48;
    v39 = v25;
    v40 = v22;
    sub_C8D5F0(&v46, v48, v27, 8);
    v26 = (unsigned int)v47;
    v25 = v39;
    v22 = v40;
    goto LABEL_35;
  }
  v38 = v47;
LABEL_37:
  v28 = (__int64 *)v46;
  v29 = v38;
  v30 = (__int64 *)sub_BD5C60(a1, v8);
  v31 = sub_B00B60(v30, v28, v29);
  v32 = sub_BD5C60(a1, v28);
  v33 = v31;
  result = (__int64 *)sub_B9F6F0(v32, v31);
  v34 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  if ( *(_QWORD *)v34 )
  {
    v35 = *(_QWORD *)(v34 + 8);
    **(_QWORD **)(v34 + 16) = v35;
    if ( v35 )
      *(_QWORD *)(v35 + 16) = *(_QWORD *)(v34 + 16);
  }
  *(_QWORD *)v34 = result;
  if ( result )
  {
    v36 = result[2];
    *(_QWORD *)(v34 + 8) = v36;
    if ( v36 )
    {
      v33 = v34 + 8;
      *(_QWORD *)(v36 + 16) = v34 + 8;
    }
    *(_QWORD *)(v34 + 16) = result + 2;
    result[2] = v34;
  }
  if ( v46 != v48 )
    return (__int64 *)_libc_free(v46, v33);
  return result;
}
