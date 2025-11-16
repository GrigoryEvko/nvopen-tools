// Function: sub_811730
// Address: 0x811730
//
__int64 __fastcall sub_811730(__int64 a1, unsigned __int8 a2, _DWORD *a3, __int64 *a4, unsigned int a5, __int64 a6)
{
  unsigned int v6; // r14d
  char v10; // r9
  __int64 v11; // r11
  __int64 v12; // r15
  __int64 v13; // rsi
  __int64 v14; // rax
  char v15; // al
  __int64 result; // rax
  _QWORD *v17; // rdi
  unsigned int v18; // r9d
  __int64 v19; // rax
  __int64 v20; // rbx
  _QWORD *v21; // rdi
  __int64 v22; // rax
  char *v23; // rdi
  __int64 v24; // rdi
  _QWORD *v25; // rax
  _QWORD *v26; // rcx
  _QWORD *v27; // rax
  unsigned __int64 v28; // r9
  _QWORD *v29; // rax
  _QWORD *v30; // rdi
  unsigned __int64 v31; // r9
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 i; // rax
  char v35; // al
  char *v36; // rsi
  __int64 v37; // rdi
  unsigned __int64 v38; // r9
  __int64 v39; // rdx
  int v40; // eax
  unsigned __int64 v41; // [rsp+8h] [rbp-88h]
  unsigned int v43; // [rsp+10h] [rbp-80h]
  _BYTE v45[112]; // [rsp+20h] [rbp-70h] BYREF

  v6 = a5;
  if ( !a5 )
  {
    v6 = unk_4D04440;
    if ( unk_4D04440 )
      v6 = sub_80A630(a1, a2);
  }
  *a4 = 0;
  *a3 = 0;
  v10 = *(_BYTE *)(a1 + 89);
  if ( a2 == 6 && *(_BYTE *)(a1 + 140) == 9 && sub_80A5F0(a1) )
  {
    v12 = a1;
    if ( (v10 & 5) != 1 )
      goto LABEL_46;
  }
  if ( (v10 & 4) == 0 )
  {
    if ( (v10 & 1) == 0 )
    {
      v12 = 0;
      goto LABEL_15;
    }
    if ( a2 != 6 )
    {
      v13 = *(_QWORD *)(a1 + 40);
      v12 = 0;
      if ( !v13 )
        goto LABEL_18;
      v15 = *(_BYTE *)(v13 + 28);
      v12 = 0;
      if ( v15 != 16 )
        goto LABEL_17;
      goto LABEL_9;
    }
LABEL_42:
    v12 = a1;
    goto LABEL_10;
  }
  v11 = *(_QWORD *)(a1 + 40);
  v12 = *(_QWORD *)(v11 + 32);
  v13 = v11;
  if ( *(_BYTE *)(v12 + 140) == 9 && sub_80A5F0(*(_QWORD *)(v11 + 32)) && (*(_BYTE *)(v12 + 89) & 5) != 1 )
  {
LABEL_46:
    v24 = *(_QWORD *)(*(_QWORD *)(v12 + 168) + 240LL);
    v25 = *(_QWORD **)(*(_QWORD *)(v24 + 152) + 168LL);
    v26 = (_QWORD *)*v25;
    if ( !*v25 )
LABEL_66:
      sub_721090();
    v27 = (_QWORD *)*v25;
    v28 = 0;
    do
    {
      v27 = (_QWORD *)*v27;
      ++v28;
    }
    while ( v27 );
    while ( 1 )
    {
      v29 = (_QWORD *)v26[7];
      if ( v29 )
        break;
LABEL_65:
      v26 = (_QWORD *)*v26;
      --v28;
      if ( !v26 )
        goto LABEL_66;
    }
    while ( v29[2] != v12 )
    {
      v29 = (_QWORD *)*v29;
      if ( !v29 )
        goto LABEL_65;
    }
    v41 = v28;
    sub_811640(v24, (_QWORD *)a6);
    v30 = (_QWORD *)qword_4F18BE0;
    ++*(_QWORD *)a6;
    v31 = v41;
    v32 = v30[2];
    if ( (unsigned __int64)(v32 + 1) > v30[1] )
    {
      sub_823810(v30);
      v30 = (_QWORD *)qword_4F18BE0;
      v31 = v41;
      v32 = *(_QWORD *)(qword_4F18BE0 + 16);
    }
    *(_BYTE *)(v30[4] + v32) = 100;
    ++v30[2];
    if ( v31 > 1 )
    {
      v38 = v31 - 2;
      if ( v38 > 9 )
      {
        v40 = sub_622470(v38, v45);
        v30 = (_QWORD *)qword_4F18BE0;
        v39 = v40;
      }
      else
      {
        v45[1] = 0;
        v45[0] = v38 + 48;
        v39 = 1;
      }
      *(_QWORD *)a6 += v39;
      sub_8238B0(v30, v45, v39);
      v30 = (_QWORD *)qword_4F18BE0;
    }
    ++*(_QWORD *)a6;
    v33 = v30[2];
    if ( (unsigned __int64)(v33 + 1) > v30[1] )
    {
      sub_823810(v30);
      v30 = (_QWORD *)qword_4F18BE0;
      v33 = *(_QWORD *)(qword_4F18BE0 + 16);
    }
    *(_BYTE *)(v30[4] + v33) = 95;
    ++v30[2];
LABEL_15:
    v13 = *(_QWORD *)(a1 + 40);
    if ( !v13 )
      goto LABEL_18;
    goto LABEL_16;
  }
  if ( (v10 & 1) == 0 )
  {
    v15 = *(_BYTE *)(v11 + 28);
    v12 = 0;
    goto LABEL_17;
  }
  v13 = v11;
  if ( a2 == 6 )
    goto LABEL_42;
LABEL_9:
  v12 = *(_QWORD *)(v13 + 32);
  if ( v12 )
  {
LABEL_10:
    v14 = sub_72B7F0(v12);
    sub_811640(v14, (_QWORD *)a6);
    if ( !sub_80AA60(v12) && a2 != 11 )
      *a4 = a1;
    goto LABEL_15;
  }
LABEL_16:
  v15 = *(_BYTE *)(v13 + 28);
LABEL_17:
  if ( v15 != 3 || (*(_BYTE *)(*(_QWORD *)(v13 + 32) + 124LL) & 0x10) == 0 )
  {
LABEL_18:
    result = v6 | (unsigned int)sub_80AFC0(a1, a2);
    if ( !(_DWORD)result )
      return result;
    v17 = (_QWORD *)qword_4F18BE0;
    ++*(_QWORD *)a6;
    v18 = a2;
    v19 = v17[2];
    if ( (unsigned __int64)(v19 + 1) > v17[1] )
    {
      sub_823810(v17);
      v17 = (_QWORD *)qword_4F18BE0;
      v18 = a2;
      v19 = *(_QWORD *)(qword_4F18BE0 + 16);
    }
    *(_BYTE *)(v17[4] + v19) = 78;
    ++v17[2];
    *a3 = 1;
    if ( a2 != 11 )
      goto LABEL_22;
    v43 = v18;
    sub_80C2B0(*(_QWORD *)(a1 + 152), (_QWORD *)a6);
    v18 = v43;
    if ( (*(_BYTE *)(a1 + 89) & 4) == 0 )
      goto LABEL_22;
    for ( i = *(_QWORD *)(a1 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v35 = *(_BYTE *)(*(_QWORD *)(i + 168) + 19LL) & 0xC0;
    if ( v35 == 64 )
    {
      v36 = "R";
    }
    else
    {
      v36 = "O";
      if ( v35 != (char)0x80 )
        goto LABEL_22;
    }
    v37 = qword_4F18BE0;
    ++*(_QWORD *)a6;
    sub_8238B0(v37, v36, 1);
    v18 = v43;
LABEL_22:
    result = sub_813790(a1, v18, v6, a4, a6);
    if ( !v12 )
    {
      *a4 = 0;
      return (__int64)a4;
    }
    return result;
  }
  if ( !v6 )
  {
LABEL_40:
    *(_QWORD *)a6 += 2LL;
    return sub_8238B0(qword_4F18BE0, "St", 2);
  }
  v20 = sub_80A110((_QWORD *)a1, a6);
  result = *(unsigned int *)(a6 + 48);
  if ( !(_DWORD)result )
  {
    v21 = (_QWORD *)qword_4F18BE0;
    ++*(_QWORD *)a6;
    v22 = v21[2];
    if ( (unsigned __int64)(v22 + 1) > v21[1] )
    {
      sub_823810(v21);
      v21 = (_QWORD *)qword_4F18BE0;
      v22 = *(_QWORD *)(qword_4F18BE0 + 16);
    }
    *(_BYTE *)(v21[4] + v22) = 78;
    ++v21[2];
    *a3 = 1;
    if ( (*(_BYTE *)(v20 + 89) & 8) != 0 )
      v23 = *(char **)(v20 + 24);
    else
      v23 = *(char **)(v20 + 8);
    sub_80BC40(v23, (_QWORD *)a6);
    goto LABEL_40;
  }
  return result;
}
