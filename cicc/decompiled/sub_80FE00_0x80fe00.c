// Function: sub_80FE00
// Address: 0x80fe00
//
__int64 __fastcall sub_80FE00(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  char v4; // al
  unsigned __int64 v5; // rdi
  _QWORD *v6; // rdi
  __int64 result; // rax
  __int64 v8; // r15
  char *v9; // r12
  size_t v10; // rax
  __int64 v11; // rdx
  size_t v12; // rax
  __int64 v13; // rax
  __int64 v14; // r13
  bool v15; // zf
  __int64 v16; // r9
  char v17; // cl
  __int64 v18; // rdi
  bool v19; // cl
  unsigned __int8 v20; // cl
  __int64 v21; // r9
  char v22; // si
  __int64 v23; // rax
  unsigned int i; // r13d
  __int64 v25; // rdx
  __int64 v26; // rdi
  _QWORD *v27; // rdi
  bool v28; // r8
  __int64 v29; // rax
  _QWORD *v30; // rdi
  __int64 v31; // rax
  char v32; // al
  _QWORD *v33; // rdi
  __int64 v34; // rax
  __int64 v35; // rdx
  _QWORD *v36; // rdi
  __int64 v37; // rax
  unsigned __int64 v38; // rdi
  __int64 v39; // rdx
  _QWORD *v40; // rdi
  __int64 v41; // rax
  __int64 v42; // r14
  int v43; // r12d
  __int64 v44; // rax
  __int64 v45; // [rsp+8h] [rbp-A8h]
  __int64 *v46; // [rsp+10h] [rbp-A0h]
  int v47[2]; // [rsp+18h] [rbp-98h]
  __int64 v48; // [rsp+20h] [rbp-90h]
  bool v49; // [rsp+2Bh] [rbp-85h]
  _BOOL4 v50; // [rsp+2Ch] [rbp-84h]
  int v51; // [rsp+38h] [rbp-78h] BYREF
  int v52; // [rsp+3Ch] [rbp-74h] BYREF
  char v53; // [rsp+40h] [rbp-70h] BYREF
  char v54; // [rsp+41h] [rbp-6Fh]

  v2 = a1;
  v4 = *(_BYTE *)(a1 + 140);
  if ( v4 != 9 )
  {
    if ( *(_QWORD *)a1 )
    {
      if ( (unsigned __int8)(v4 - 9) > 2u )
        goto LABEL_4;
      v8 = *(_QWORD *)(a1 + 168);
LABEL_13:
      if ( !*(_BYTE *)(v8 + 113) )
      {
LABEL_4:
        *(_QWORD *)a2 += 2LL;
        sub_8238B0(qword_4F18BE0, "Ut", 2);
        if ( (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) > 2u )
        {
          v5 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 96LL) + 16LL);
          if ( v5 <= 1 )
          {
LABEL_7:
            v6 = (_QWORD *)qword_4F18BE0;
            ++*(_QWORD *)a2;
            result = v6[2];
            if ( (unsigned __int64)(result + 1) > v6[1] )
            {
              sub_823810(v6);
              v6 = (_QWORD *)qword_4F18BE0;
              result = *(_QWORD *)(qword_4F18BE0 + 16);
            }
            *(_BYTE *)(v6[4] + result) = 95;
            ++v6[2];
            return result;
          }
LABEL_6:
          sub_80BEC0(v5, 0, (_QWORD *)a2);
          goto LABEL_7;
        }
LABEL_21:
        v5 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v2 + 96LL) + 168LL);
        if ( v5 <= 1 )
          goto LABEL_7;
        goto LABEL_6;
      }
    }
LABEL_14:
    v9 = sub_80AEF0(a1);
    v10 = strlen(v9);
    if ( v10 > 9 )
    {
      v11 = (int)sub_622470(v10, &v53);
    }
    else
    {
      v54 = 0;
      v11 = 1;
      v53 = v10 + 48;
    }
    *(_QWORD *)a2 += v11;
    sub_8238B0(qword_4F18BE0, &v53, v11);
    v12 = strlen(v9);
    *(_QWORD *)a2 += v12;
    return sub_8238B0(qword_4F18BE0, v9, v12);
  }
  v8 = *(_QWORD *)(a1 + 168);
  if ( (*(_BYTE *)(v8 + 109) & 0x20) == 0 )
  {
    if ( *(_QWORD *)a1 )
      goto LABEL_13;
    goto LABEL_14;
  }
  v15 = *(_QWORD *)(v8 + 96) == 0;
  *(_DWORD *)(a2 + 64) = *(_QWORD *)(v8 + 96) != 0;
  if ( v15 || !*(_DWORD *)(a2 + 68) )
  {
    v13 = sub_80A8B0(*(_QWORD *)a1, 0);
    *(_QWORD *)a2 += 2LL;
    v14 = v13;
    sub_8238B0(qword_4F18BE0, "Ul", 2);
    if ( (*(_BYTE *)(v14 - 8) & 8) != 0 )
      v14 = *(_QWORD *)(v14 + 176);
    sub_80FC70(*(_QWORD *)(v14 + 168), (_QWORD *)a2);
    ++*(_QWORD *)a2;
    sub_8238B0(qword_4F18BE0, "E", 1);
    if ( *(_BYTE *)(a1 + 140) == 12 )
    {
      do
        v2 = *(_QWORD *)(v2 + 160);
      while ( *(_BYTE *)(v2 + 140) == 12 );
    }
    goto LABEL_21;
  }
  v16 = *(_QWORD *)(v8 + 96);
  v17 = *(_BYTE *)(v8 + 92);
  v51 = 0;
  v18 = *(_QWORD *)a1;
  v19 = (v17 & 4) != 0;
  v48 = *(_QWORD *)(v16 + 128);
  v50 = v19;
  if ( v19 )
  {
    v44 = sub_80A8B0(v18, &v51);
    v52 = 0;
    v46 = (__int64 *)v44;
    v45 = 0;
  }
  else
  {
    v52 = 0;
    v46 = 0;
    v45 = sub_80A8B0(v18, &v52);
  }
  if ( !dword_4F06978 )
    v51 = 0;
  v22 = *(_BYTE *)(v8 + 92);
  if ( *(_BYTE *)(v21 + 173) == 7 )
    *(_QWORD *)v47 = *(_QWORD *)(v21 + 200);
  else
    *(_QWORD *)v47 = *(_QWORD *)(v21 + 184);
  v23 = *(_QWORD *)(v2 + 160);
  for ( i = 0; v23; ++i )
    v23 = *(_QWORD *)(v23 + 112);
  v25 = *(_QWORD *)a2;
  v26 = qword_4F18BE0;
  if ( (v20 & ((*(_BYTE *)(v8 + 92) & 0x10) == 0)) != 0 )
  {
    --i;
LABEL_35:
    v49 = (*(_BYTE *)(v8 + 92) & 0x10) != 0;
    *(_QWORD *)a2 = v25 + 6;
    sub_8238B0(v26, "Unvhdl", 6);
    ++*(_QWORD *)a2;
    v53 = ((v22 & 8) != 0) + 48;
    v54 = 0;
    sub_8238B0(qword_4F18BE0, &v53, 1);
    v27 = (_QWORD *)qword_4F18BE0;
    ++*(_QWORD *)a2;
    v28 = v49;
    v29 = v27[2];
    if ( (unsigned __int64)(v29 + 1) > v27[1] )
    {
      sub_823810(v27);
      v27 = (_QWORD *)qword_4F18BE0;
      v28 = v49;
      v29 = *(_QWORD *)(qword_4F18BE0 + 16);
    }
    *(_BYTE *)(v27[4] + v29) = 95;
    ++v27[2];
    ++*(_QWORD *)a2;
    v53 = v28 + 48;
    v54 = 0;
    sub_8238B0(v27, &v53, 1);
    v30 = (_QWORD *)qword_4F18BE0;
    ++*(_QWORD *)a2;
    v31 = v30[2];
    if ( (unsigned __int64)(v31 + 1) > v30[1] )
    {
      sub_823810(v30);
      v30 = (_QWORD *)qword_4F18BE0;
      v31 = *(_QWORD *)(qword_4F18BE0 + 16);
    }
    *(_BYTE *)(v30[4] + v31) = 95;
    ++v30[2];
    v32 = 48 - ((v51 == 0) - 1);
    ++*(_QWORD *)a2;
    v53 = v32;
    v54 = 0;
    sub_8238B0(v30, &v53, 1);
    v33 = (_QWORD *)qword_4F18BE0;
    ++*(_QWORD *)a2;
    v34 = v33[2];
    if ( (unsigned __int64)(v34 + 1) > v33[1] )
    {
      sub_823810(v33);
      v33 = (_QWORD *)qword_4F18BE0;
      v34 = *(_QWORD *)(qword_4F18BE0 + 16);
    }
    *(_BYTE *)(v33[4] + v34) = 95;
    ++v33[2];
    goto LABEL_42;
  }
  if ( v50 )
    goto LABEL_35;
  if ( (*(_BYTE *)(v8 + 92) & 2) != 0 )
  {
    *(_QWORD *)a2 = v25 + 6;
    sub_8238B0(v26, "Unvdtl", 6);
  }
  else
  {
    *(_QWORD *)a2 = v25 + 5;
    sub_8238B0(v26, "Unvdl", 5);
  }
LABEL_42:
  if ( i > 9 )
  {
    v35 = (int)sub_622470(i, &v53);
  }
  else
  {
    v54 = 0;
    v35 = 1;
    v53 = i + 48;
  }
  *(_QWORD *)a2 += v35;
  sub_8238B0(qword_4F18BE0, &v53, v35);
  v36 = (_QWORD *)qword_4F18BE0;
  ++*(_QWORD *)a2;
  v37 = v36[2];
  if ( (unsigned __int64)(v37 + 1) > v36[1] )
  {
    sub_823810(v36);
    v36 = (_QWORD *)qword_4F18BE0;
    v37 = *(_QWORD *)(qword_4F18BE0 + 16);
  }
  *(_BYTE *)(v36[4] + v37) = 95;
  ++v36[2];
  sub_80F5E0(v48, 0, (_QWORD *)a2);
  sub_8111C0(v47[0], a2);
  if ( (*(_BYTE *)(v8 + 92) & 2) != 0 )
    sub_80F5E0(*(_QWORD *)(v45 + 160), 0, (_QWORD *)a2);
  v38 = *(unsigned int *)(v8 + 104);
  if ( (unsigned int)v38 > 9 )
  {
    v39 = (int)sub_622470(v38, &v53);
  }
  else
  {
    v54 = 0;
    v39 = 1;
    v53 = v38 + 48;
  }
  *(_QWORD *)a2 += v39;
  sub_8238B0(qword_4F18BE0, &v53, v39);
  v40 = (_QWORD *)qword_4F18BE0;
  ++*(_QWORD *)a2;
  v41 = v40[2];
  if ( (unsigned __int64)(v41 + 1) > v40[1] )
  {
    sub_823810(v40);
    v40 = (_QWORD *)qword_4F18BE0;
    v41 = *(_QWORD *)(qword_4F18BE0 + 16);
  }
  *(_BYTE *)(v40[4] + v41) = 95;
  result = v50;
  ++v40[2];
  if ( v50 )
  {
    if ( (*(_BYTE *)(v46 - 1) & 8) != 0 )
      v46 = (__int64 *)v46[22];
    sub_80F5E0(v46[20], 0, (_QWORD *)a2);
    sub_80FC70(v46[21], (_QWORD *)a2);
    ++*(_QWORD *)a2;
    result = sub_8238B0(qword_4F18BE0, "E", 1);
  }
  v42 = *(_QWORD *)(v2 + 160);
  v43 = 0;
  if ( i )
  {
    do
    {
      ++v43;
      result = sub_80F5E0(*(_QWORD *)(v42 + 120), 0, (_QWORD *)a2);
      v42 = *(_QWORD *)(v42 + 112);
    }
    while ( i != v43 );
  }
  return result;
}
