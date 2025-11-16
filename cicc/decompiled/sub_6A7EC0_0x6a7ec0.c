// Function: sub_6A7EC0
// Address: 0x6a7ec0
//
__int64 __fastcall sub_6A7EC0(_WORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rsi
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r15
  __int64 v13; // rdi
  unsigned int v14; // ebx
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rbx
  bool v19; // cf
  bool v20; // zf
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // rbx
  unsigned int v24; // esi
  __int64 v25; // rax
  __int64 v26; // rbx
  int v27; // esi
  __int64 v28; // rbx
  unsigned __int8 v29; // di
  int v30; // [rsp+18h] [rbp-298h]
  __int16 v31; // [rsp+1Eh] [rbp-292h]
  int v32; // [rsp+28h] [rbp-288h]
  _BOOL4 v33; // [rsp+2Ch] [rbp-284h]
  unsigned int v34; // [rsp+38h] [rbp-278h] BYREF
  int v35; // [rsp+3Ch] [rbp-274h] BYREF
  int v36; // [rsp+40h] [rbp-270h] BYREF
  unsigned int v37; // [rsp+44h] [rbp-26Ch] BYREF
  __int64 v38; // [rsp+48h] [rbp-268h] BYREF
  __int64 v39; // [rsp+50h] [rbp-260h] BYREF
  __int64 v40; // [rsp+58h] [rbp-258h] BYREF
  __int64 v41; // [rsp+60h] [rbp-250h] BYREF
  __int64 v42; // [rsp+68h] [rbp-248h] BYREF
  __int64 v43; // [rsp+70h] [rbp-240h] BYREF
  __int64 v44; // [rsp+78h] [rbp-238h] BYREF
  _BYTE v45[160]; // [rsp+80h] [rbp-230h] BYREF
  _QWORD v46[2]; // [rsp+120h] [rbp-190h] BYREF
  char v47; // [rsp+130h] [rbp-180h]
  __int64 v48; // [rsp+164h] [rbp-14Ch] BYREF
  int v49; // [rsp+16Ch] [rbp-144h]
  __int16 v50; // [rsp+170h] [rbp-140h]
  __int64 v51; // [rsp+1B0h] [rbp-100h]

  v34 = 0;
  v38 = sub_724DC0(a1, a2, a3, a4, a5, a6);
  if ( a1 )
  {
    v33 = a1[4] == 247;
    sub_6F8810(
      (_DWORD)a1,
      (unsigned int)&v34,
      (unsigned int)v46,
      (unsigned int)&v39,
      (unsigned int)&v41,
      (unsigned int)v45,
      (__int64)&v43);
    v32 = v34 == 0;
    v31 = *(_WORD *)(*(_QWORD *)a1 + 48LL);
    v30 = *(_DWORD *)(*(_QWORD *)a1 + 44LL);
    v42 = v41;
    sub_68B050((unsigned int)dword_4F04C64, (__int64)&v37, &v40);
    v8 = (__int64)v45;
    v9 = 5;
    sub_6E2140(5, v45, 0, 0, a1);
    *(_BYTE *)(qword_4D03C50 + 18LL) |= 0x20u;
    if ( dword_4F077C4 == 2 || unk_4F07778 <= 202310 )
      goto LABEL_4;
LABEL_34:
    if ( qword_4D04A00 )
    {
      v19 = *(_QWORD *)(qword_4D04A00 + 16) < 8u;
      v20 = *(_QWORD *)(qword_4D04A00 + 16) == 8;
      if ( *(_QWORD *)(qword_4D04A00 + 16) == 8 )
      {
        v8 = *(_QWORD *)(qword_4D04A00 + 8);
        v11 = 8;
        v9 = (__int64)"_Alignof";
        do
        {
          if ( !v11 )
            break;
          v19 = *(_BYTE *)v8 < *(_BYTE *)v9;
          v20 = *(_BYTE *)v8++ == *(_BYTE *)v9++;
          --v11;
        }
        while ( v20 );
        if ( (!v19 && !v20) == v19 )
        {
          v9 = dword_4F063F8;
          if ( !(unsigned int)sub_729F80(dword_4F063F8) )
          {
            sub_684AA0(4 - ((dword_4D04964 == 0) - 1), 0xCDAu, &dword_4F063F8);
            v8 = 1;
            v9 = 3290;
            sub_67D850(3290, 1, 0);
          }
        }
      }
    }
    if ( a1 )
      goto LABEL_4;
    goto LABEL_37;
  }
  v33 = word_4F06418[0] == 247;
  v41 = *(_QWORD *)&dword_4F063F8;
  v42 = *(_QWORD *)&dword_4F063F8;
  sub_68B050((unsigned int)dword_4F04C64, (__int64)&v37, &v40);
  v8 = (__int64)v45;
  v9 = 5;
  sub_6E2140(5, v45, 0, 0, 0);
  *(_BYTE *)(qword_4D03C50 + 18LL) |= 0x20u;
  v32 = 0;
  if ( dword_4F077C4 != 2 && unk_4F07778 > 202310 )
    goto LABEL_34;
LABEL_37:
  sub_7B8B50(v9, v8, v10, v11);
  if ( word_4F06418[0] != 27 )
  {
    if ( v34 )
    {
      v43 = *(_QWORD *)&dword_4F063F8;
      goto LABEL_47;
    }
    sub_69ED20((__int64)v46, 0, 18, 0);
    goto LABEL_40;
  }
  v44 = *(_QWORD *)&dword_4F063F8;
  sub_7B8B50(v9, v8, v21, v22);
  if ( (unsigned int)sub_679C10(5u) )
  {
    v34 = 1;
    v43 = *(_QWORD *)&dword_4F063F8;
  }
  else
  {
    if ( !v34 )
    {
      sub_69ED20((__int64)v46, 0, 18, 8);
      v48 = v44;
LABEL_40:
      v32 = 1;
      v30 = v49;
      v31 = v50;
      goto LABEL_4;
    }
    v43 = *(_QWORD *)&dword_4F063F8;
  }
  ++*(_BYTE *)(qword_4F061C8 + 36LL);
  ++*(_QWORD *)(qword_4D03C50 + 40LL);
  sub_65CD60(&v39);
  v30 = qword_4F063F0;
  v31 = WORD2(qword_4F063F0);
  sub_7BE280(28, 18, 0, 0);
  --*(_BYTE *)(qword_4F061C8 + 36LL);
  --*(_QWORD *)(qword_4D03C50 + 40LL);
  if ( dword_4D0477C && word_4F06418[0] == 73 )
  {
    sub_68D9C0((__int64)&v39, (__int64)&v44, &v43, 0, 0, a2, 0);
    v39 = *(_QWORD *)a2;
  }
LABEL_4:
  if ( v34 )
  {
LABEL_47:
    if ( !(unsigned int)sub_8D32E0(v39)
      || (dword_4F04C44 != -1
       || (v25 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v25 + 6) & 6) != 0)
       || *(_BYTE *)(v25 + 4) == 12)
      && (unsigned int)sub_8DBE70(v39) )
    {
      v13 = v39;
      v12 = 0;
    }
    else
    {
      v12 = 0;
      v39 = sub_8D46C0(v39);
      v13 = v39;
    }
LABEL_53:
    if ( !v33 )
      goto LABEL_7;
    goto LABEL_17;
  }
  if ( !v33 )
  {
    sub_6F69D0(v46, 39);
    v12 = 0;
    sub_831BB0(v46);
    v13 = v46[0];
    v39 = v46[0];
    v43 = v48;
    if ( v47 != 1 )
      goto LABEL_7;
    goto LABEL_55;
  }
  v16 = 5;
  if ( dword_4D04964 )
    v16 = unk_4F07471;
  sub_6E5C80(v16, 2471, &v48);
  sub_6F69D0(v46, 39);
  v12 = 0;
  sub_831BB0(v46);
  v13 = v46[0];
  v39 = v46[0];
  v43 = v48;
  if ( v47 == 1 )
  {
LABEL_55:
    v12 = v51;
    goto LABEL_53;
  }
LABEL_17:
  if ( (unsigned int)sub_8D2310(v13) && (!HIDWORD(qword_4F077B4) || (_DWORD)qword_4F077B4) )
  {
    sub_6E5D20(7, 2961, &v43, v39);
    v13 = v39;
    LOBYTE(v33) = 1;
  }
  else
  {
    v13 = v39;
    LOBYTE(v33) = 1;
    if ( dword_4F077C4 != 2 && unk_4F07778 > 201111 )
    {
      if ( (unsigned int)sub_8D23E0(v39) )
      {
        v29 = 5;
        if ( dword_4D04964 )
          v29 = unk_4F07471;
        sub_684AA0(v29, 0xB92u, &v43);
        v13 = v39;
        LOBYTE(v33) = 1;
      }
      else
      {
        v13 = v39;
      }
    }
  }
LABEL_7:
  v14 = sub_731F80(v13, v34, v12, &v42, &v36, &v35);
  if ( !v34 )
  {
    if ( !v35 )
    {
      if ( dword_4F04C44 == -1 )
      {
        v15 = qword_4F04C68[0] + 776LL * dword_4F04C64;
        if ( (*(_BYTE *)(v15 + 6) & 6) == 0 && *(_BYTE *)(v15 + 4) != 12 )
        {
          if ( v36 )
          {
LABEL_13:
            sub_72C970(v38);
            goto LABEL_29;
          }
LABEL_28:
          sub_72BBE0(v38, v14, unk_4F06A51);
          if ( *(_BYTE *)(qword_4D03C50 + 16LL) )
          {
            sub_729730(v37);
            v27 = v34;
            if ( !v34 && unk_4F07270 == unk_4F073B8 && (unk_4F04C50 || (v27 = dword_4F04C38) != 0) )
            {
              v34 = 1;
              v27 = 1;
            }
            v28 = v38;
            *(_QWORD *)(v28 + 144) = sub_688D10(14, v27, v39, v46, 0);
            LODWORD(v28) = v34;
            *(_BYTE *)(*(_QWORD *)(v38 + 144) + 57LL) = v33;
            sub_7296F0((unsigned int)dword_4F04C64, &v37);
            v32 = ((_DWORD)v28 != 0) & (unsigned __int8)v32;
          }
          goto LABEL_29;
        }
      }
      if ( !(unsigned int)sub_696840((__int64)v46) )
        goto LABEL_26;
      v35 = 1;
    }
    if ( v36 )
      goto LABEL_13;
    goto LABEL_43;
  }
LABEL_26:
  if ( v36 )
    goto LABEL_13;
  if ( !v35 )
    goto LABEL_28;
LABEL_43:
  sub_724C70(v38, 12);
  sub_7249B0(v38, 7);
  v23 = v38;
  v24 = v34;
  *(_QWORD *)(v38 + 184) = v39;
  *(_BYTE *)(v23 + 200) = v33 | *(_BYTE *)(v23 + 200) & 0xFE;
  if ( !v24 )
  {
    sub_6F40C0(v46);
    v26 = v38;
    v32 = 0;
    *(_QWORD *)(v26 + 192) = sub_6F6F40(v46, 0);
    v23 = v38;
  }
  *(_QWORD *)(v23 + 128) = sub_72BA30(unk_4F06A51);
LABEL_29:
  sub_6E6A50(v38, a2);
  if ( v32 )
    sub_6E24C0();
  *(_DWORD *)(a2 + 68) = v42;
  *(_WORD *)(a2 + 72) = WORD2(v42);
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(a2 + 68);
  *(_DWORD *)(a2 + 76) = v30;
  *(_WORD *)(a2 + 80) = v31;
  unk_4F061D8 = *(_QWORD *)(a2 + 76);
  sub_6E3280(a2, &v41);
  sub_6E3BA0(a2, &v41, 0, &v43);
  sub_6E2B30(a2, &v41);
  v17 = v40;
  sub_729730(v37);
  qword_4F06BC0 = v17;
  return sub_724E30(&v38);
}
