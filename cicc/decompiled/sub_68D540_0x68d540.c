// Function: sub_68D540
// Address: 0x68d540
//
const __m128i *__fastcall sub_68D540(
        _QWORD *a1,
        __int64 a2,
        int a3,
        unsigned int a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        const __m128i *a9)
{
  __int64 v13; // rcx
  __int64 v14; // rsi
  char v15; // al
  __int64 v16; // rax
  char i; // dl
  const __m128i *result; // rax
  __int64 v19; // r13
  int v20; // r15d
  __int64 v21; // r13
  unsigned int v22; // r8d
  __int64 v23; // r9
  unsigned __int8 v24; // si
  int v25; // eax
  __int64 v26; // rcx
  _BOOL4 v27; // r8d
  __int64 v28; // rsi
  int v29; // eax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rdi
  unsigned int v35; // eax
  int v36; // eax
  int v37; // eax
  char v38; // al
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rax
  int v43; // eax
  __int64 v44; // rax
  __int64 v45; // [rsp+0h] [rbp-50h]
  __int64 v46; // [rsp+0h] [rbp-50h]
  __int64 v47; // [rsp+0h] [rbp-50h]
  __int64 v48; // [rsp+0h] [rbp-50h]
  __int64 v49; // [rsp+0h] [rbp-50h]
  __int64 v50; // [rsp+0h] [rbp-50h]
  __int64 v51; // [rsp+8h] [rbp-48h]
  _BOOL4 v52; // [rsp+8h] [rbp-48h]
  unsigned int v53; // [rsp+8h] [rbp-48h]
  __int64 v54; // [rsp+8h] [rbp-48h]
  __int64 v55; // [rsp+8h] [rbp-48h]
  __int64 v56; // [rsp+8h] [rbp-48h]
  unsigned int v57; // [rsp+8h] [rbp-48h]
  unsigned int v59; // [rsp+8h] [rbp-48h]
  unsigned int v60; // [rsp+8h] [rbp-48h]
  _BOOL4 v61; // [rsp+8h] [rbp-48h]
  __int64 v62; // [rsp+8h] [rbp-48h]
  _DWORD v63[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v13 = a6;
  v14 = *(_QWORD *)(a5 + 24);
  v63[0] = a3;
  v15 = *(_BYTE *)(v14 + 80);
  if ( v15 == 16 )
  {
    v14 = **(_QWORD **)(v14 + 88);
    v15 = *(_BYTE *)(v14 + 80);
  }
  if ( v15 == 24 )
    v14 = *(_QWORD *)(v14 + 88);
  if ( !*((_BYTE *)a1 + 16) )
    return (const __m128i *)sub_6E6260(a9);
  v16 = *a1;
  for ( i = *(_BYTE *)(*a1 + 140LL); i == 12; i = *(_BYTE *)(v16 + 140) )
    v16 = *(_QWORD *)(v16 + 160);
  if ( !i )
    return (const __m128i *)sub_6E6260(a9);
  v19 = *(_QWORD *)(v14 + 88);
  if ( unk_4D044D0 )
  {
    if ( a2 )
    {
      if ( (*(_BYTE *)(a2 + 140) & 0xFB) == 8 )
      {
        v38 = sub_8D4C10(a2, dword_4F077C4 != 2);
        v13 = a6;
        if ( (v38 & 8) != 0 )
        {
          sub_684B30(0xADBu, (_DWORD *)a1 + 17);
          v13 = a6;
        }
      }
    }
  }
  if ( v63[0] || *((_BYTE *)a1 + 17) == 1 && (v56 = v13, v36 = sub_6ED0A0(a1), v13 = v56, !v36) )
  {
    if ( (*(_BYTE *)(a2 + 140) & 0xFB) != 8 )
    {
      v20 = HIDWORD(qword_4D0495C);
      if ( HIDWORD(qword_4D0495C) )
      {
        v21 = *(_QWORD *)(v19 + 120);
        v20 = 0;
        v22 = 1;
        v23 = v21;
      }
      else
      {
        v62 = v13;
        v44 = sub_73CB50(v19, 0);
        v13 = v62;
        v22 = 1;
        v21 = v44;
        v23 = v44;
      }
      goto LABEL_18;
    }
    v34 = a2;
    v55 = v13;
    v20 = 0;
    v35 = sub_8D4C10(v34, dword_4F077C4 != 2);
    v13 = v55;
    v22 = 1;
    v28 = v35;
  }
  else
  {
    v51 = v13;
    v25 = sub_6ED0A0(a1);
    v26 = v51;
    v27 = (unk_4D04410 | v25) != 0;
    LODWORD(v28) = 0;
    if ( (*(_BYTE *)(a2 + 140) & 0xFB) == 8 )
    {
      v50 = v51;
      v61 = (unk_4D04410 | v25) != 0;
      v43 = sub_8D4C10(a2, dword_4F077C4 != 2);
      v26 = v50;
      v27 = v61;
      LODWORD(v28) = v43;
    }
    v45 = v26;
    v52 = v27;
    v29 = sub_8D3D40(*a1);
    v22 = v52;
    v13 = v45;
    if ( v29 )
    {
      v20 = v52;
      v21 = *(_QWORD *)&dword_4D03B80;
      goto LABEL_34;
    }
    v28 = (unsigned int)v28;
    v20 = v52;
  }
  if ( HIDWORD(qword_4D0495C) )
  {
    v21 = *(_QWORD *)(v19 + 120);
    if ( (_DWORD)v28 )
    {
      if ( *((_BYTE *)a1 + 17) == 2 || (v47 = v13, v57 = v22, v37 = sub_6ED0A0(a1), v22 = v57, v13 = v47, v37) )
      {
        if ( !v63[0] )
        {
          v48 = v13;
          v59 = v22;
          sub_6FFCF0(a1, v63);
          v39 = sub_8D46C0(*a1);
          v40 = sub_73D4C0(v39, dword_4F077C4 == 2);
          v41 = sub_72D2E0(v40, 0);
          sub_6FC3F0(v41, a1, 1);
          sub_6F9270(a1);
          sub_6FA3A0(a1);
          v22 = v59;
          v63[0] = 0;
          v13 = v48;
          if ( v59 )
          {
            v23 = v21;
            v24 = 94;
            goto LABEL_19;
          }
          goto LABEL_35;
        }
      }
      v46 = v13;
      v53 = v22;
      sub_6FFCF0(a1, v63);
      v30 = sub_8D46C0(*a1);
      v31 = sub_73D4C0(v30, dword_4F077C4 == 2);
      v32 = sub_72D2E0(v31, 0);
      sub_6FC3F0(v32, a1, 1);
      v22 = v53;
      v13 = v46;
    }
  }
  else
  {
    v49 = v13;
    v60 = v22;
    v42 = sub_73CB50(v19, v28);
    v13 = v49;
    v22 = v60;
    v21 = v42;
  }
LABEL_34:
  v23 = v21;
  if ( !v22 )
  {
LABEL_35:
    v54 = v13;
    v33 = sub_73D720(v21);
    v23 = v21;
    v13 = v54;
    v22 = 0;
    v21 = v33;
  }
LABEL_18:
  v24 = 94 - ((v63[0] == 0) - 1);
LABEL_19:
  sub_68D1C0((__int64)a1, v24, a5, v13, a7, v23, v22, a4, a9);
  if ( dword_4F077C4 == 2 && (unsigned int)sub_8D32E0(v21) )
    return (const __m128i *)sub_6F82C0(a9);
  result = (const __m128i *)a1[11];
  a9[5].m128i_i64[1] = (__int64)result;
  if ( a8 )
  {
    *(_QWORD *)(a8 + 48) = result;
    result = a9;
    a9[5].m128i_i64[1] = a8;
  }
  if ( v20 )
    return (const __m128i *)sub_6ED1A0(a9);
  return result;
}
