// Function: sub_7B1C00
// Address: 0x7b1c00
//
_DWORD *__fastcall sub_7B1C00(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        char *a4,
        int a5,
        char a6,
        char a7,
        char a8,
        int a9,
        int a10,
        __int64 a11,
        __int64 a12)
{
  int v15; // eax
  int v16; // r14d
  int v17; // eax
  __int64 v18; // rbx
  int v19; // edx
  _QWORD *v20; // rax
  _QWORD *v21; // rbx
  FILE *v22; // rax
  char *v23; // rax
  _QWORD *v24; // rcx
  _QWORD *v25; // rax
  __int64 v26; // rcx
  unsigned int v27; // esi
  __int64 v28; // rdi
  _QWORD *v29; // rbx
  __int64 v30; // rdi
  int v31; // ebx
  _DWORD *result; // rax
  const char *v33; // rax
  const char *v34; // rax
  int v35; // ebx
  __int64 v36; // rax
  __int64 v37; // rdx
  const char *v38; // rax
  char v39; // [rsp+Ch] [rbp-54h]
  unsigned __int8 v43; // [rsp+27h] [rbp-39h]

  v15 = sub_7ABED0(a4);
  if ( v15 > 9 )
    sub_685220(3u, (__int64)a4);
  v16 = v15;
  if ( unk_4D0493C )
    sub_7B1260();
  if ( qword_4D04908 && byte_4F17F98 )
    sub_7AFA40((__int64)a4, a2);
  dword_4F04D98 = unk_4F04D9C;
  v17 = dword_4F17FD8;
  if ( dword_4F17FD8 + 1 != dword_4F17FDC )
  {
    v18 = (__int64)qword_4F064B0;
    goto LABEL_9;
  }
  v35 = dword_4F17FD8 + 31;
  v36 = sub_822C60(qword_4F17FE0, 112LL * v35 - 3360, 112LL * v35);
  dword_4F17FDC = v35;
  qword_4F17FE0 = v36;
  v37 = v36;
  v17 = dword_4F17FD8;
  if ( dword_4F17FD8 >= 0 )
  {
    v18 = v37 + 112LL * dword_4F17FD8;
    qword_4F064B0 = (_QWORD *)v18;
    qword_4F17FD0 = v18;
LABEL_9:
    if ( v17 > 7 )
    {
      *(_QWORD *)(v18 + 48) = ftell(*(FILE **)v18);
      fclose((FILE *)*qword_4F064B0);
      *qword_4F064B0 = 0;
      v17 = dword_4F17FD8;
      v18 = (__int64)qword_4F064B0;
    }
    goto LABEL_11;
  }
  v18 = (__int64)qword_4F064B0;
LABEL_11:
  if ( !a5 )
  {
    v43 = 0;
    LOBYTE(v19) = 0;
    goto LABEL_13;
  }
  if ( (*(_BYTE *)(v18 + 88) & 2) != 0 )
    goto LABEL_36;
  if ( !a11 )
  {
    v43 = 0;
    LOBYTE(v19) = 0;
    goto LABEL_37;
  }
  v43 = 0;
  v19 = *(_DWORD *)(a11 + 8);
  if ( v19 )
  {
LABEL_36:
    v43 = 1;
    LOBYTE(v19) = 1;
  }
LABEL_37:
  ++unk_4F064AC;
LABEL_13:
  v39 = v19;
  dword_4F17FD8 = v17 + 1;
  dword_4F17FC0 = 0;
  v20 = (_QWORD *)(qword_4F17FE0 + 112LL * (v17 + 1));
  qword_4F064B0 = v20;
  *v20 = a1;
  v21 = qword_4F064B0;
  qword_4F17FD0 = (__int64)v20;
  v22 = (FILE *)*qword_4F064B0;
  *((_DWORD *)qword_4F064B0 + 10) = 0;
  v21[6] = 0;
  qword_4F17FC8 = v22;
  *((_DWORD *)v21 + 20) = 0;
  v21[2] = a4;
  v21[1] = a3;
  v23 = sub_722430(a4, 1);
  v24 = qword_4F064B0;
  v21[3] = v23;
  v24[4] = a11;
  v24[12] = a12;
  *((_WORD *)v24 + 44) = v24[11] & 0xC0
                       | (32 * (a8 & 1))
                       | (16 * (a7 & 1))
                       | (8 * (dword_4F064B8[0] & 1))
                       | (4 * (v16 != 0))
                       | a5 & 1
                       | (2 * v43);
  *((_DWORD *)v24 + 26) = a10;
  unk_4F064A8 = a10;
  sub_722830((__int64)&unk_4F17FB0, a10);
  if ( unk_4F064A8 )
    dword_4D0432C = 1;
  v25 = qword_4F064B0;
  dword_4F064B8[0] = 0;
  v26 = dword_4F17FD8;
  *((_BYTE *)qword_4F064B0 + 88) &= ~0x40u;
  v27 = unk_4F06468;
  if ( (_DWORD)v26 )
  {
    v28 = *(_QWORD *)(112 * v26 + qword_4F17FE0 - 56);
    if ( !a9 )
    {
      v27 = unk_4F06468 + 1;
      goto LABEL_18;
    }
  }
  else
  {
    if ( !a9 )
    {
      v27 = unk_4F06468 + 1;
      v28 = 0;
      goto LABEL_18;
    }
    dword_4F17FA8 = 0;
    v28 = qword_4F07280;
  }
  --unk_4F06468;
LABEL_18:
  sub_729880(v28, v27, 1u, a3, (__int64)a4, a2, v25 + 7, a5, a6, a7, a8, a9, v39);
  v29 = qword_4F064B0;
  v30 = qword_4F064B0[7];
  qword_4F064B0[8] = v30;
  *((_DWORD *)v29 + 21) = sub_67D0F0(v30);
  if ( unk_4D0493C )
  {
    if ( dword_4F17FD8 )
      sub_7AF280(49, 1);
    else
      sub_7AF280(32, 1);
  }
  if ( qword_4D04908 )
  {
    if ( dword_4F17FD8 )
    {
      sub_7AF3F0(49);
      if ( !(_DWORD)qword_4D04914 )
        goto LABEL_25;
LABEL_43:
      v33 = (const char *)sub_723260(unk_4F076B8);
      fprintf(qword_4D04928, "%s:", v33);
      v34 = (const char *)sub_723260((char *)qword_4F064B0[1]);
      fprintf(qword_4D04928, " %s\n", v34);
      goto LABEL_25;
    }
    sub_7AF3F0(32);
  }
  if ( (_DWORD)qword_4D04914 )
    goto LABEL_43;
LABEL_25:
  if ( HIDWORD(qword_4D04914) )
  {
    v31 = dword_4F17FD8;
    if ( dword_4F17FD8 )
    {
      v38 = (const char *)sub_723260((char *)qword_4F064B0[1]);
      fprintf(qword_4F07510, "%*s%s\n", v31 - 1, byte_3F871B3, v38);
    }
  }
  if ( *(char *)(qword_4F064B0[8] + 72LL) >= 0 )
    sub_720AB0(qword_4F064B0[3], (qword_4F064B0[11] & 2) != 0);
  result = &dword_4F077C4;
  if ( dword_4F077C4 != 1 )
  {
    result = (_DWORD *)unk_4D03CD8;
    unk_4D03CD0 = unk_4D03CD8;
    qword_4F064B0[9] = unk_4D03CD8;
  }
  return result;
}
