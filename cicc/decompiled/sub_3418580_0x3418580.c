// Function: sub_3418580
// Address: 0x3418580
//
__int64 __fastcall sub_3418580(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  void *v4; // rdx
  __int64 v5; // rdi
  __int64 v6; // rdi
  _BYTE *v7; // rax
  _BYTE *v8; // rax
  __int64 v9; // rcx
  int *v10; // r13
  int *v11; // r12
  unsigned int v12; // eax
  __int64 v13; // rdx
  bool v14; // cc
  _WORD *v15; // rax
  __int64 v16; // rax
  unsigned __int8 v17; // dl
  __int64 v18; // rax
  __int64 v19; // rdi
  unsigned __int8 *v20; // rax
  size_t v21; // rdx
  _BYTE *v22; // rdi
  __int64 v24; // r11
  __int64 v25; // rsi
  __int64 v26; // rdx
  void (__fastcall *v27)(__int64 *, __int64 *, __int64); // rax
  unsigned __int64 v28; // rdx
  __int64 v29; // r8
  __int64 v30; // r8
  _BYTE *v31; // rax
  __int64 v32; // rax
  __int64 v33; // [rsp+8h] [rbp-58h]
  __int64 v34; // [rsp+8h] [rbp-58h]
  size_t v35; // [rsp+8h] [rbp-58h]
  __int64 v36[2]; // [rsp+10h] [rbp-50h] BYREF
  void (__fastcall *v37)(__int64 *, __int64 *, __int64); // [rsp+20h] [rbp-40h]
  __int64 (__fastcall *v38)(unsigned __int64 *, __int64); // [rsp+28h] [rbp-38h]

  v2 = a2;
  v4 = *(void **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v4 <= 0xDu )
  {
    v5 = sub_CB6200(a2, " DbgVal(Order=", 0xEu);
  }
  else
  {
    v5 = a2;
    qmemcpy(v4, " DbgVal(Order=", 14);
    *(_QWORD *)(a2 + 32) += 14LL;
  }
  v6 = sub_CB59D0(v5, *(unsigned int *)(a1 + 56));
  v7 = *(_BYTE **)(v6 + 32);
  if ( (unsigned __int64)v7 >= *(_QWORD *)(v6 + 24) )
  {
    sub_CB5D20(v6, 41);
  }
  else
  {
    *(_QWORD *)(v6 + 32) = v7 + 1;
    *v7 = 41;
  }
  v8 = *(_BYTE **)(a2 + 32);
  if ( *(_BYTE *)(a1 + 62) )
  {
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v8 <= 0xCu )
    {
      sub_CB6200(a2, "(Invalidated)", 0xDu);
      v8 = *(_BYTE **)(a2 + 32);
    }
    else
    {
      qmemcpy(v8, "(Invalidated)", 13);
      v8 = (_BYTE *)(*(_QWORD *)(a2 + 32) + 13LL);
      *(_QWORD *)(a2 + 32) = v8;
    }
  }
  if ( *(_BYTE *)(a1 + 63) )
  {
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v8 <= 8u )
    {
      sub_CB6200(a2, "(Emitted)", 9u);
      v8 = *(_BYTE **)(a2 + 32);
    }
    else
    {
      v8[8] = 41;
      *(_QWORD *)v8 = 0x64657474696D4528LL;
      v8 = (_BYTE *)(*(_QWORD *)(a2 + 32) + 9LL);
      *(_QWORD *)(a2 + 32) = v8;
    }
  }
  if ( *(_BYTE **)(a2 + 24) == v8 )
  {
    sub_CB6200(a2, (unsigned __int8 *)"(", 1u);
    v9 = *(_QWORD *)(a2 + 32);
  }
  else
  {
    *v8 = 40;
    v9 = *(_QWORD *)(a2 + 32) + 1LL;
    *(_QWORD *)(a2 + 32) = v9;
  }
  v10 = *(int **)(a1 + 8);
  v11 = &v10[6 * *(_QWORD *)a1];
  if ( v10 == v11 )
    goto LABEL_21;
LABEL_10:
  v12 = *v10;
  v13 = *(_QWORD *)(v2 + 24);
  v14 = (unsigned int)*v10 <= 2;
  if ( *v10 != 2 )
  {
LABEL_11:
    if ( v14 )
    {
      if ( v12 )
      {
        if ( (unsigned __int64)(v13 - v9) <= 4 )
        {
          sub_CB6200(v2, (unsigned __int8 *)"CONST", 5u);
          v9 = *(_QWORD *)(v2 + 32);
        }
        else
        {
          *(_DWORD *)v9 = 1397641027;
          *(_BYTE *)(v9 + 4) = 84;
          v9 = *(_QWORD *)(v2 + 32) + 5LL;
          *(_QWORD *)(v2 + 32) = v9;
        }
        goto LABEL_15;
      }
      v28 = v13 - v9;
      if ( !*((_QWORD *)v10 + 1) )
      {
        if ( v28 <= 5 )
        {
          sub_CB6200(v2, "SDNODE", 6u);
          v9 = *(_QWORD *)(v2 + 32);
        }
        else
        {
          *(_DWORD *)v9 = 1330529363;
          *(_WORD *)(v9 + 4) = 17732;
          v9 = *(_QWORD *)(v2 + 32) + 6LL;
          *(_QWORD *)(v2 + 32) = v9;
        }
LABEL_15:
        v10 += 6;
        if ( v11 != v10 )
          goto LABEL_16;
        goto LABEL_21;
      }
      if ( v28 <= 6 )
      {
        v29 = sub_CB6200(v2, "SDNODE=", 7u);
      }
      else
      {
        *(_DWORD *)v9 = 1330529363;
        *(_WORD *)(v9 + 4) = 17732;
        v29 = v2;
        *(_BYTE *)(v9 + 6) = 61;
        *(_QWORD *)(v2 + 32) += 7LL;
      }
      v34 = v29;
      v36[0] = *((_QWORD *)v10 + 1);
      v37 = (void (__fastcall *)(__int64 *, __int64 *, __int64))sub_34181C0;
      v38 = sub_34181B0;
      sub_34181B0((unsigned __int64 *)v36, v29);
      v30 = v34;
      v31 = *(_BYTE **)(v34 + 32);
      if ( (unsigned __int64)v31 >= *(_QWORD *)(v34 + 24) )
      {
        v30 = sub_CB5D20(v34, 58);
      }
      else
      {
        *(_QWORD *)(v34 + 32) = v31 + 1;
        *v31 = 58;
      }
      sub_CB59D0(v30, (unsigned int)v10[4]);
      v27 = v37;
      if ( !v37 )
      {
LABEL_40:
        v9 = *(_QWORD *)(v2 + 32);
        goto LABEL_15;
      }
    }
    else
    {
      if ( v12 != 3 )
        goto LABEL_15;
      if ( (unsigned __int64)(v13 - v9) <= 4 )
      {
        v24 = sub_CB6200(v2, "VREG=", 5u);
      }
      else
      {
        *(_DWORD *)v9 = 1195725398;
        v24 = v2;
        *(_BYTE *)(v9 + 4) = 61;
        *(_QWORD *)(v2 + 32) += 5LL;
      }
      v25 = (unsigned int)v10[2];
      v33 = v24;
      sub_2FF6320(v36, v25, 0, 0, 0);
      if ( !v37 )
        sub_4263D6(v36, v25, v26);
      v38((unsigned __int64 *)v36, v33);
      v27 = v37;
      if ( !v37 )
        goto LABEL_40;
    }
    v27(v36, v36, 3);
    goto LABEL_40;
  }
  while ( 1 )
  {
    if ( (unsigned __int64)(v13 - v9) <= 7 )
    {
      v32 = sub_CB6200(v2, "FRAMEIX=", 8u);
      sub_CB59D0(v32, (unsigned int)v10[2]);
    }
    else
    {
      *(_QWORD *)v9 = 0x3D5849454D415246LL;
      *(_QWORD *)(v2 + 32) += 8LL;
      sub_CB59D0(v2, (unsigned int)v10[2]);
    }
    v9 = *(_QWORD *)(v2 + 32);
    v10 += 6;
    if ( v11 == v10 )
      break;
LABEL_16:
    if ( (unsigned __int64)(*(_QWORD *)(v2 + 24) - v9) <= 1 )
    {
      sub_CB6200(v2, (unsigned __int8 *)", ", 2u);
      v9 = *(_QWORD *)(v2 + 32);
      goto LABEL_10;
    }
    *(_WORD *)v9 = 8236;
    v13 = *(_QWORD *)(v2 + 24);
    v9 = *(_QWORD *)(v2 + 32) + 2LL;
    *(_QWORD *)(v2 + 32) = v9;
    v12 = *v10;
    v14 = (unsigned int)*v10 <= 2;
    if ( *v10 != 2 )
      goto LABEL_11;
  }
LABEL_21:
  if ( *(_QWORD *)(v2 + 24) == v9 )
  {
    sub_CB6200(v2, (unsigned __int8 *)")", 1u);
    v15 = *(_WORD **)(v2 + 32);
  }
  else
  {
    *(_BYTE *)v9 = 41;
    v15 = (_WORD *)(*(_QWORD *)(v2 + 32) + 1LL);
    *(_QWORD *)(v2 + 32) = v15;
  }
  if ( *(_BYTE *)(a1 + 60) )
  {
    if ( *(_QWORD *)(v2 + 24) - (_QWORD)v15 <= 9u )
    {
      sub_CB6200(v2, "(Indirect)", 0xAu);
      v15 = *(_WORD **)(v2 + 32);
    }
    else
    {
      qmemcpy(v15, "(Indirect)", 10);
      v15 = (_WORD *)(*(_QWORD *)(v2 + 32) + 10LL);
      *(_QWORD *)(v2 + 32) = v15;
    }
  }
  if ( *(_BYTE *)(a1 + 61) )
  {
    if ( *(_QWORD *)(v2 + 24) - (_QWORD)v15 <= 9u )
    {
      sub_CB6200(v2, "(Variadic)", 0xAu);
      v15 = *(_WORD **)(v2 + 32);
    }
    else
    {
      qmemcpy(v15, "(Variadic)", 10);
      v15 = (_WORD *)(*(_QWORD *)(v2 + 32) + 10LL);
      *(_QWORD *)(v2 + 32) = v15;
    }
  }
  if ( *(_QWORD *)(v2 + 24) - (_QWORD)v15 <= 1u )
  {
    v2 = sub_CB6200(v2, (unsigned __int8 *)":\"", 2u);
    v16 = *(_QWORD *)(a1 + 32);
    v17 = *(_BYTE *)(v16 - 16);
    if ( (v17 & 2) == 0 )
      goto LABEL_27;
  }
  else
  {
    *v15 = 8762;
    *(_QWORD *)(v2 + 32) += 2LL;
    v16 = *(_QWORD *)(a1 + 32);
    v17 = *(_BYTE *)(v16 - 16);
    if ( (v17 & 2) == 0 )
    {
LABEL_27:
      v18 = v16 - 16 - 8LL * ((v17 >> 2) & 0xF);
      goto LABEL_28;
    }
  }
  v18 = *(_QWORD *)(v16 - 32);
LABEL_28:
  v19 = *(_QWORD *)(v18 + 8);
  if ( !v19 )
  {
LABEL_31:
    v22 = *(_BYTE **)(v2 + 32);
    goto LABEL_32;
  }
  v20 = (unsigned __int8 *)sub_B91420(v19);
  v22 = *(_BYTE **)(v2 + 32);
  if ( v21 > *(_QWORD *)(v2 + 24) - (_QWORD)v22 )
  {
    v2 = sub_CB6200(v2, v20, v21);
    goto LABEL_31;
  }
  if ( v21 )
  {
    v35 = v21;
    memcpy(v22, v20, v21);
    v22 = (_BYTE *)(v35 + *(_QWORD *)(v2 + 32));
    *(_QWORD *)(v2 + 32) = v22;
  }
LABEL_32:
  if ( *(_QWORD *)(v2 + 24) <= (unsigned __int64)v22 )
    return sub_CB5D20(v2, 34);
  *(_QWORD *)(v2 + 32) = v22 + 1;
  *v22 = 34;
  return (__int64)(v22 + 1);
}
