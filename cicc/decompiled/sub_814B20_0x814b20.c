// Function: sub_814B20
// Address: 0x814b20
//
unsigned __int8 *__fastcall sub_814B20(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r15
  _QWORD *v8; // rdi
  __int64 v9; // rax
  bool v10; // zf
  unsigned __int64 v11; // rsi
  unsigned __int64 v12; // rcx
  _QWORD *v13; // r12
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 v16; // rax
  const char *v17; // r15
  size_t v18; // rax
  __int64 v20; // rcx
  __int64 v21; // rdi
  __int64 v22; // rax
  __int64 v23; // rbx
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30[4]; // [rsp+0h] [rbp-80h] BYREF
  char v31; // [rsp+20h] [rbp-60h]
  __int64 v32; // [rsp+28h] [rbp-58h]
  __int64 v33; // [rsp+30h] [rbp-50h]
  int v34; // [rsp+38h] [rbp-48h]
  char v35; // [rsp+3Ch] [rbp-44h]
  __int64 v36; // [rsp+40h] [rbp-40h]

  v30[0] = 0;
  v7 = a1[34];
  v30[3] = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v34 = 0;
  v35 = 0;
  v36 = 0;
  sub_809110(a1, a2, a3, a4, a5, a6, 0, 0, 0);
  sub_823800(qword_4F18BE0);
  v30[0] += 2;
  sub_8238B0(qword_4F18BE0, &unk_3C1BC40, 2);
  if ( a1[38] || a1[39] )
  {
    v30[0] += 2;
    sub_8238B0(qword_4F18BE0, &unk_3C1BC3C, 2);
    v8 = (_QWORD *)qword_4F18BE0;
    v9 = *(_QWORD *)(qword_4F18BE0 + 16);
  }
  else
  {
    v8 = (_QWORD *)qword_4F18BE0;
    ++v30[0];
    v26 = *(_QWORD *)(qword_4F18BE0 + 16);
    if ( (unsigned __int64)(v26 + 1) > *(_QWORD *)(qword_4F18BE0 + 8) )
    {
      sub_823810(qword_4F18BE0);
      v8 = (_QWORD *)qword_4F18BE0;
      v26 = *(_QWORD *)(qword_4F18BE0 + 16);
    }
    *(_BYTE *)(v8[4] + v26) = 84;
    v9 = v8[2] + 1LL;
    v8[2] = v9;
  }
  v10 = a1[37] == 0;
  v11 = v9 + 1;
  v12 = v8[1];
  ++v30[0];
  if ( v10 )
  {
    if ( v11 > v12 )
    {
      sub_823810(v8);
      v8 = (_QWORD *)qword_4F18BE0;
      *(_BYTE *)(*(_QWORD *)(qword_4F18BE0 + 32) + *(_QWORD *)(qword_4F18BE0 + 16)) = 104;
    }
    else
    {
      *(_BYTE *)(v8[4] + v9) = 104;
    }
    ++v8[2];
  }
  else
  {
    if ( v11 > v12 )
    {
      sub_823810(v8);
      v8 = (_QWORD *)qword_4F18BE0;
      v9 = *(_QWORD *)(qword_4F18BE0 + 16);
    }
    *(_BYTE *)(v8[4] + v9) = 118;
    ++v8[2];
  }
  sub_809180(a1[36], v30);
  v13 = (_QWORD *)qword_4F18BE0;
  ++v30[0];
  v14 = *(_QWORD *)(qword_4F18BE0 + 16);
  if ( (unsigned __int64)(v14 + 1) > *(_QWORD *)(qword_4F18BE0 + 8) )
  {
    sub_823810(qword_4F18BE0);
    v13 = (_QWORD *)qword_4F18BE0;
    v14 = *(_QWORD *)(qword_4F18BE0 + 16);
  }
  *(_BYTE *)(v13[4] + v14) = 95;
  ++v13[2];
  v15 = a1[37];
  if ( v15 )
  {
    v28 = sub_7E1340();
    sub_809180(v15 * v28, v30);
    v13 = (_QWORD *)qword_4F18BE0;
    ++v30[0];
    v29 = *(_QWORD *)(qword_4F18BE0 + 16);
    if ( (unsigned __int64)(v29 + 1) > *(_QWORD *)(qword_4F18BE0 + 8) )
    {
      sub_823810(qword_4F18BE0);
      v13 = (_QWORD *)qword_4F18BE0;
      v29 = *(_QWORD *)(qword_4F18BE0 + 16);
    }
    *(_BYTE *)(v13[4] + v29) = 95;
    ++v13[2];
  }
  v16 = a1[39];
  if ( !a1[38] )
  {
    if ( !v16 )
      goto LABEL_13;
    goto LABEL_33;
  }
  v20 = v13[2];
  v21 = (__int64)v13;
  if ( v16 )
  {
LABEL_33:
    v27 = v13[2];
    ++v30[0];
    if ( (unsigned __int64)(v27 + 1) > v13[1] )
    {
      sub_823810(v13);
      v13 = (_QWORD *)qword_4F18BE0;
      v27 = *(_QWORD *)(qword_4F18BE0 + 16);
    }
    *(_BYTE *)(v13[4] + v27) = 118;
    ++v13[2];
    goto LABEL_20;
  }
  ++v30[0];
  if ( (unsigned __int64)(v20 + 1) > v13[1] )
  {
    sub_823810(v13);
    v21 = qword_4F18BE0;
    v20 = *(_QWORD *)(qword_4F18BE0 + 16);
  }
  *(_BYTE *)(*(_QWORD *)(v21 + 32) + v20) = 104;
  ++*(_QWORD *)(v21 + 16);
LABEL_20:
  sub_809180(a1[38], v30);
  v13 = (_QWORD *)qword_4F18BE0;
  ++v30[0];
  v22 = *(_QWORD *)(qword_4F18BE0 + 16);
  if ( (unsigned __int64)(v22 + 1) > *(_QWORD *)(qword_4F18BE0 + 8) )
  {
    sub_823810(qword_4F18BE0);
    v13 = (_QWORD *)qword_4F18BE0;
    v22 = *(_QWORD *)(qword_4F18BE0 + 16);
  }
  *(_BYTE *)(v13[4] + v22) = 95;
  ++v13[2];
  v23 = a1[39];
  if ( v23 )
  {
    v24 = sub_7E1340();
    sub_809180(v23 * v24, v30);
    v13 = (_QWORD *)qword_4F18BE0;
    ++v30[0];
    v25 = *(_QWORD *)(qword_4F18BE0 + 16);
    if ( (unsigned __int64)(v25 + 1) > *(_QWORD *)(qword_4F18BE0 + 8) )
    {
      sub_823810(qword_4F18BE0);
      v13 = (_QWORD *)qword_4F18BE0;
      v25 = *(_QWORD *)(qword_4F18BE0 + 16);
    }
    *(_BYTE *)(v13[4] + v25) = 95;
    ++v13[2];
    if ( (*(_BYTE *)(v7 + 89) & 8) != 0 )
      goto LABEL_14;
LABEL_26:
    sub_8111C0(v7, 0, 0, 0, 0, 0, (__int64)v30);
    return sub_80B290((__int64)a1, 1, (__int64)v30);
  }
LABEL_13:
  if ( (*(_BYTE *)(v7 + 89) & 8) == 0 )
    goto LABEL_26;
LABEL_14:
  v17 = (const char *)(*(_QWORD *)(v7 + 8) + 2LL);
  v18 = strlen(v17);
  v30[0] += v18;
  sub_8238B0(v13, v17, v18);
  return sub_80B290((__int64)a1, 1, (__int64)v30);
}
