// Function: sub_7F3E10
// Address: 0x7f3e10
//
void __fastcall sub_7F3E10(const __m128i *a1)
{
  __m128i *v2; // r15
  __m128i *v3; // rdi
  __int8 v4; // bl
  const __m128i **v5; // r12
  const __m128i *v6; // rdi
  __int64 v7; // rdi
  __int64 *v8; // r13
  __int64 v9; // r12
  _QWORD *i; // rcx
  __m128i *v11; // rax
  __m128i *v12; // r12
  __int64 v13; // rax
  __m128i *v14; // rdi
  __int64 v15; // rax
  __int64 v16; // r12
  __int64 v17; // rdi
  const __m128i *v18; // rdi
  __int64 v19; // rdi
  char v20; // al
  __int64 v21; // rax
  __m128i *v22; // r13
  _QWORD *v23; // rax
  void *v24; // rax
  __int64 v25; // rbx
  _QWORD *v26; // rdi
  __m128i *v27; // rax
  __m128i *v28; // r15
  const __m128i **v29; // r13
  __int64 v30; // rdi
  __m128i *v31; // rax
  _QWORD *j; // rcx
  __m128i *v33; // rbx
  _BYTE *v34; // [rsp+8h] [rbp-158h]
  const __m128i *v35; // [rsp+18h] [rbp-148h]
  __int64 v36; // [rsp+20h] [rbp-140h]
  __int64 v37; // [rsp+20h] [rbp-140h]
  _BYTE *v38; // [rsp+20h] [rbp-140h]
  __m128i *v39[2]; // [rsp+28h] [rbp-138h] BYREF
  __m128i *v40; // [rsp+38h] [rbp-128h] BYREF
  int v41[8]; // [rsp+40h] [rbp-120h] BYREF
  _BYTE v42[32]; // [rsp+60h] [rbp-100h] BYREF
  _BYTE v43[80]; // [rsp+80h] [rbp-E0h] BYREF
  _BYTE v44[144]; // [rsp+D0h] [rbp-90h] BYREF

  v2 = (__m128i *)a1;
  v39[0] = (__m128i *)a1;
  v3 = (__m128i *)a1[3].m128i_i64[0];
  v4 = a1[2].m128i_i8[8];
  if ( !v3 )
  {
    if ( v4 != 16 )
      goto LABEL_5;
    goto LABEL_39;
  }
  if ( v3[1].m128i_i8[8] != 9 )
  {
    if ( v4 != 16 )
    {
      sub_7F2A70(v3, 1);
      v2 = v39[0];
LABEL_5:
      v5 = (const __m128i **)v2[4].m128i_i64[1];
      if ( (unsigned __int8)(v4 - 3) <= 1u || v4 == 1 )
      {
        sub_7EC960((const __m128i *)v2[4].m128i_i64[1]);
        sub_7EC960((const __m128i *)v2[5].m128i_i64[0]);
        return;
      }
      if ( v4 == 2 )
      {
        sub_7EC960(*v5);
        sub_7EC960(v5[1]);
        return;
      }
      v6 = (const __m128i *)v2[4].m128i_i64[1];
      if ( v4 != 5 )
      {
        sub_7EC960(v6);
        v7 = *(_QWORD *)(v39[0][5].m128i_i64[0] + 8);
        if ( v7 )
          sub_7F2600(v7, 0);
        return;
      }
LABEL_40:
      sub_7EC960(v6);
      return;
    }
LABEL_39:
    sub_7F2600((__int64)v3, 0);
    v6 = (const __m128i *)v39[0][4].m128i_i64[1];
    goto LABEL_40;
  }
  v8 = (__int64 *)v3[3].m128i_i64[1];
  if ( (v4 & 0xF7) == 5 )
  {
    v40 = (__m128i *)a1[4].m128i_i64[1];
    v2 = v40;
    v9 = *v8;
    sub_7E7090(v40, (__int64)v41, &v40);
    if ( (v2[2].m128i_i8[9] & 1) != 0 )
    {
      for ( i = 0; ; i[3] = v40 )
      {
        i = sub_732D20((__int64)v2, v9, 0, i);
        if ( !i )
          break;
      }
      v11 = v40;
      v2[2].m128i_i8[9] &= ~1u;
      v11[2].m128i_i8[9] |= 1u;
    }
    v12 = v39[0];
    v13 = v39[0][1].m128i_i64[0];
    if ( !v13 || *(_BYTE *)(v13 + 40) != 7 || (v37 = *(_QWORD *)(v13 + 72), (*(_BYTE *)(v37 + 120) & 2) == 0) )
    {
      v36 = qword_4D03F68[1];
      sub_7E7090(v39[0], (__int64)v42, v39);
      if ( (v12[2].m128i_i8[9] & 1) != 0 )
      {
        for ( j = 0; ; j[3] = v39[0] )
        {
          j = sub_732D20((__int64)v12, v36, 0, j);
          if ( !j )
            break;
        }
        v14 = v39[0];
        v12[2].m128i_i8[9] &= ~1u;
        v14[2].m128i_i8[9] |= 1u;
      }
      else
      {
        v14 = v39[0];
      }
      sub_7E1720((__int64)v14, (__int64)v42);
      v15 = sub_806990(v42);
      *(_BYTE *)(v15 + 120) |= 2u;
      v37 = v15;
    }
  }
  else
  {
    sub_7E7090(a1, (__int64)v41, v39);
    v37 = 0;
  }
  v16 = *v8;
  *(_QWORD *)(v2[5].m128i_i64[0] + 8) = *v8;
  sub_726E40(v16, 2, 0);
  *(_QWORD *)(v16 + 80) = v2;
  sub_7E18E0((__int64)v44, v16, 0);
  v17 = *(_QWORD *)(v16 + 88);
  if ( v17 )
    sub_7E9190(v17, (__int64)v41);
  v18 = (const __m128i *)v8[3];
  if ( v18 )
  {
    sub_7EC960(v18);
    v19 = v8[3];
    v20 = *(_BYTE *)(v19 + 40);
    if ( v20 == 20 )
    {
      v34 = sub_726B30(11);
      sub_7E6810((__int64)v34, (__int64)v41, 1);
      sub_7E1740((__int64)v34, (__int64)v43);
      sub_7E6930(*(__int64 **)(v8[3] + 72), (__int64)v43);
    }
    else
    {
      if ( v20 != 11 && v20 )
        sub_721090();
      sub_7E6810(v19, (__int64)v41, 1);
    }
  }
  v21 = v8[1];
  if ( v21 )
  {
    sub_7F9080(*(_QWORD *)(v21 + 8), v43);
    sub_7FEC50(v8[1], (unsigned int)v43, 0, 0, 1, 0, (__int64)v41, 0, 0);
    if ( unk_4D03EB0 )
      sub_7FAF20(v2);
  }
  v22 = (__m128i *)v8[2];
  if ( v4 == 16 )
  {
    sub_7F2600((__int64)v22, 0);
    v31 = v39[0];
    v39[0][3].m128i_i64[0] = (__int64)v22;
    sub_7EC960((const __m128i *)v31[4].m128i_i64[1]);
  }
  else
  {
    sub_7F2A70(v22, 1);
    if ( (v4 & 0xF7) == 5 )
    {
      v35 = (const __m128i *)sub_726B30(6);
      v35[4].m128i_i64[1] = v37;
      v35[5].m128i_i64[0] = *(_QWORD *)(*(_QWORD *)(v37 + 128) + 80LL);
      v38 = sub_726B30(1);
      v23 = sub_72BA30(5u);
      v24 = sub_73DBF0(0x1Du, (__int64)v23, (__int64)v22);
      *((_QWORD *)v38 + 9) = v35;
      *((_QWORD *)v38 + 6) = v24;
      sub_7E6810((__int64)v38, (__int64)v41, 1);
      sub_7E7540(v35);
      sub_7EC960(v40);
      sub_7E1720((__int64)v40, (__int64)v41);
      if ( v4 == 13 )
      {
        v25 = v39[0][5].m128i_i64[0];
        v39[0][3].m128i_i64[0] = 0;
        v26 = *(_QWORD **)(v25 + 8);
        if ( v26 )
        {
          v27 = (__m128i *)sub_7E69E0(v26, v41);
          sub_7F2600(*(_QWORD *)(v25 + 8), v27);
          *(_QWORD *)(v25 + 8) = 0;
          if ( unk_4D03EB0 )
            sub_7FAF20(v2);
        }
      }
      else
      {
        v33 = v39[0];
        v33[3].m128i_i64[0] = (__int64)sub_73A830(1, 5u);
      }
      goto LABEL_47;
    }
    v28 = v39[0];
    v39[0][3].m128i_i64[0] = (__int64)v22;
    v29 = (const __m128i **)v28[4].m128i_i64[1];
    if ( v4 == 1 )
    {
      sub_7EC960((const __m128i *)v28[4].m128i_i64[1]);
      sub_7EC960((const __m128i *)v28[5].m128i_i64[0]);
    }
    else
    {
      sub_7EC960(*v29);
      sub_7EC960(v29[1]);
    }
  }
  sub_7E1720((__int64)v39[0]->m128i_i64, (__int64)v41);
LABEL_47:
  v30 = *(_QWORD *)(v16 + 88);
  if ( v30 )
    sub_7E7530(v30, (__int64)v41);
  sub_7E1AA0();
}
