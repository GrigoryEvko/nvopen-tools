// Function: sub_6FED50
// Address: 0x6fed50
//
void __fastcall sub_6FED50(__int64 a1, int a2, int a3, __int64 a4, __int64 a5, _QWORD *a6)
{
  _DWORD *v6; // r10
  _DWORD *v8; // r13
  __int64 v10; // rsi
  int v11; // r12d
  unsigned int v13; // edi
  __int64 v14; // rax
  __int64 v15; // rdx
  _DWORD *v16; // rsi
  __int64 v17; // rdi
  __int64 v18; // r8
  __int64 v19; // r9
  _QWORD *v20; // rcx
  char v21; // al
  __int64 v22; // rdx
  bool v23; // r13
  __int64 v24; // rdi
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // rax
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r10
  int v37; // eax
  _QWORD *v38; // rax
  __int64 v39; // rbx
  __int64 v40; // rax
  __int64 v41; // rdx
  int v42; // eax
  __int64 v43; // rax
  __int64 v44; // rdi
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rdi
  __int64 v52; // r13
  bool v53; // zf
  __int64 *v54; // rax
  __int64 v55; // rax
  __int64 v56; // [rsp+0h] [rbp-1C0h]
  __int64 v57; // [rsp+8h] [rbp-1B8h]
  __int64 v58; // [rsp+10h] [rbp-1B0h]
  __int64 v59; // [rsp+10h] [rbp-1B0h]
  int v60; // [rsp+18h] [rbp-1A8h]
  char v62; // [rsp+1Ch] [rbp-1A4h]
  __int64 v63; // [rsp+28h] [rbp-198h] BYREF
  _OWORD v64[9]; // [rsp+30h] [rbp-190h] BYREF
  __m128i v65; // [rsp+C0h] [rbp-100h]
  __m128i v66; // [rsp+D0h] [rbp-F0h]
  __m128i v67; // [rsp+E0h] [rbp-E0h]
  __m128i v68; // [rsp+F0h] [rbp-D0h]
  __m128i v69; // [rsp+100h] [rbp-C0h]
  __m128i v70; // [rsp+110h] [rbp-B0h]
  __m128i v71; // [rsp+120h] [rbp-A0h]
  __m128i v72; // [rsp+130h] [rbp-90h]
  __m128i v73; // [rsp+140h] [rbp-80h]
  __m128i v74; // [rsp+150h] [rbp-70h]
  __m128i v75; // [rsp+160h] [rbp-60h]
  __m128i v76; // [rsp+170h] [rbp-50h]
  __m128i v77; // [rsp+180h] [rbp-40h]

  v6 = (_DWORD *)(a1 + 68);
  v8 = a6;
  if ( !a6 )
    v8 = (_DWORD *)(a1 + 68);
  if ( !*(_BYTE *)(a1 + 16) )
    goto LABEL_8;
  v10 = *(_QWORD *)a1;
  v11 = a5;
  v13 = *(unsigned __int8 *)(*(_QWORD *)a1 + 140LL);
  if ( (_BYTE)v13 == 12 )
  {
    v14 = v10;
    do
    {
      v14 = *(_QWORD *)(v14 + 160);
      v15 = *(unsigned __int8 *)(v14 + 140);
    }
    while ( (_BYTE)v15 == 12 );
  }
  else
  {
    v15 = v13;
  }
  if ( !(_BYTE)v15 )
  {
LABEL_8:
    sub_6E6870(a1);
    sub_6E5820(*(unsigned __int64 **)(a1 + 88), 32);
    goto LABEL_9;
  }
  if ( *(_BYTE *)(a1 + 17) == 3 )
  {
    while ( (_BYTE)v13 == 12 )
    {
      v10 = *(_QWORD *)(v10 + 160);
      LOBYTE(v13) = *(_BYTE *)(v10 + 140);
    }
    if ( (_BYTE)v13 == 7 && (*(_BYTE *)(*(_QWORD *)(v10 + 168) + 20LL) & 2) != 0 )
      sub_6E5470(*(_QWORD *)(v10 + 104), v6);
  }
  v16 = v8;
  v17 = a1;
  v60 = sub_6ECEA0(a1, v8, v15, a4, a5, (__int64)a6);
  if ( !v60 )
  {
    v64[0] = _mm_loadu_si128((const __m128i *)a1);
    v64[1] = _mm_loadu_si128((const __m128i *)(a1 + 16));
    v20 = &qword_4D03C50;
    v64[2] = _mm_loadu_si128((const __m128i *)(a1 + 32));
    v21 = *(_BYTE *)(a1 + 16);
    v64[3] = _mm_loadu_si128((const __m128i *)(a1 + 48));
    v22 = qword_4D03C50;
    v64[4] = _mm_loadu_si128((const __m128i *)(a1 + 64));
    v64[5] = _mm_loadu_si128((const __m128i *)(a1 + 80));
    v64[6] = _mm_loadu_si128((const __m128i *)(a1 + 96));
    v64[7] = _mm_loadu_si128((const __m128i *)(a1 + 112));
    v64[8] = _mm_loadu_si128((const __m128i *)(a1 + 128));
    switch ( v21 )
    {
      case 2:
        v65 = _mm_loadu_si128((const __m128i *)(a1 + 144));
        v66 = _mm_loadu_si128((const __m128i *)(a1 + 160));
        v67 = _mm_loadu_si128((const __m128i *)(a1 + 176));
        v68 = _mm_loadu_si128((const __m128i *)(a1 + 192));
        v69 = _mm_loadu_si128((const __m128i *)(a1 + 208));
        v70 = _mm_loadu_si128((const __m128i *)(a1 + 224));
        v71 = _mm_loadu_si128((const __m128i *)(a1 + 240));
        v72 = _mm_loadu_si128((const __m128i *)(a1 + 256));
        v73 = _mm_loadu_si128((const __m128i *)(a1 + 272));
        v74 = _mm_loadu_si128((const __m128i *)(a1 + 288));
        v75 = _mm_loadu_si128((const __m128i *)(a1 + 304));
        v76 = _mm_loadu_si128((const __m128i *)(a1 + 320));
        v77 = _mm_loadu_si128((const __m128i *)(a1 + 336));
        break;
      case 5:
        v65.m128i_i64[0] = *(_QWORD *)(a1 + 144);
        break;
      case 1:
        v65.m128i_i64[0] = *(_QWORD *)(a1 + 144);
        if ( (*(_BYTE *)(qword_4D03C50 + 18LL) & 8) == 0 )
          goto LABEL_16;
        goto LABEL_43;
    }
    if ( (*(_BYTE *)(qword_4D03C50 + 18LL) & 8) == 0 )
    {
LABEL_16:
      v57 = 0;
      goto LABEL_17;
    }
    v17 = a1;
    if ( !sub_694910((_BYTE *)a1) )
    {
      if ( *(_BYTE *)(a1 + 16) != 2 )
      {
        v22 = qword_4D03C50;
        goto LABEL_16;
      }
      v23 = *(_BYTE *)(a1 + 317) == 12;
      LOBYTE(v22) = *(_BYTE *)(a1 + 317) != 12;
      v60 = v23;
      if ( *(_BYTE *)(qword_4D03C50 + 16LL) || *(_BYTE *)(a1 + 317) == 12 )
        goto LABEL_81;
LABEL_27:
      sub_6E4EE0(a1, (__int64)v64);
      sub_6E5820(*(unsigned __int64 **)(a1 + 88), 32);
      goto LABEL_9;
    }
LABEL_43:
    v63 = sub_724DC0(v17, v8, v22, v20, v18, v19);
    v33 = sub_6F6F40((const __m128i *)a1, 0, v29, v30, v31, v32);
    v57 = v33;
    if ( *(_BYTE *)(v33 + 24) == 1 )
    {
      v36 = v33;
      while ( *(_BYTE *)(v36 + 56) == 6 )
      {
        v38 = *(_QWORD **)(v36 + 72);
        v58 = v36;
        if ( *(_QWORD *)v36 == *v38 )
        {
          v36 = *(_QWORD *)(v36 + 72);
        }
        else
        {
          v37 = sub_8D97D0(*(_QWORD *)v36, *v38, 0, v34, v35);
          v36 = v58;
          if ( !v37 )
            break;
          v36 = *(_QWORD *)(v58 + 72);
        }
        if ( *(_BYTE *)(v36 + 24) != 1 )
          goto LABEL_77;
      }
      if ( *(_BYTE *)(qword_4D03C50 + 16LL) > 3u )
        goto LABEL_69;
      if ( *(_BYTE *)(v36 + 24) == 1 && *(_BYTE *)(v36 + 56) == 95 )
      {
        v47 = *(_QWORD *)(v36 + 72);
        if ( *(_BYTE *)(v47 + 24) == 2 && *(_BYTE *)(*(_QWORD *)(v47 + 56) + 173LL) == 12 )
        {
LABEL_93:
          v16 = (_DWORD *)v63;
          if ( !(unsigned int)sub_717510(v36, v63, 1) )
          {
LABEL_94:
            sub_724E30(&v63);
            v22 = 0;
            v60 = 1;
            goto LABEL_19;
          }
          v60 = 1;
LABEL_70:
          v44 = v63;
          if ( unk_4F07734 && *(_BYTE *)(v63 + 173) == 12 && *(_BYTE *)(v63 + 176) == 4 && a6 )
            goto LABEL_94;
          if ( !a2 )
          {
LABEL_73:
            v16 = (_DWORD *)a1;
            sub_6E6A50(v44, a1);
            sub_724E30(&v63);
            LOBYTE(v22) = (v60 ^ 1) & 1;
            if ( *(_BYTE *)(qword_4D03C50 + 16LL) )
            {
              v23 = 0;
              goto LABEL_20;
            }
            goto LABEL_27;
          }
          if ( (unsigned int)sub_8D2E30(*(_QWORD *)(v63 + 128)) )
          {
            v51 = sub_8D46C0(*(_QWORD *)(v63 + 128));
            if ( a3 )
              v52 = sub_72D6A0(v51);
            else
              v52 = sub_72D600(v51);
            v44 = v63;
            v53 = *(_BYTE *)(v63 + 173) == 12;
            *(_QWORD *)(v63 + 128) = v52;
            if ( !v53 || *(_BYTE *)(v44 + 176) != 1 )
              goto LABEL_73;
            v54 = (__int64 *)sub_72E9A0(v44);
            *v54 = v52;
            v55 = sub_6E4240((__int64)v54, 0);
            if ( *(_BYTE *)(v55 + 24) == 2 )
              *(_QWORD *)(*(_QWORD *)(v55 + 56) + 128LL) = v52;
          }
          v44 = v63;
          goto LABEL_73;
        }
      }
    }
    else
    {
      v36 = v33;
LABEL_77:
      if ( *(_BYTE *)(qword_4D03C50 + 16LL) > 3u )
        goto LABEL_69;
    }
    if ( dword_4F04C44 != -1
      || (v46 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v46 + 6) & 6) != 0)
      || *(_BYTE *)(v46 + 4) == 12 )
    {
      v41 = v57;
      if ( *(_BYTE *)(v57 + 24) == 1 && *(_BYTE *)(v57 + 56) == 7 )
        v41 = *(_QWORD *)(v57 + 72);
      v59 = v36;
      v56 = v41;
      v42 = sub_82F8B0(v41);
      v36 = v59;
      if ( v42 )
      {
        v43 = *(_QWORD *)(v56 + 72);
        if ( *(_BYTE *)(v43 + 24) == 2 )
        {
          v48 = *(_QWORD *)(v43 + 16);
          if ( *(_BYTE *)(v48 + 24) == 1 && *(_BYTE *)(v48 + 56) == 116 )
          {
            v49 = *(_QWORD *)(v48 + 72);
            if ( *(_BYTE *)(v49 + 24) == 2 )
            {
              v50 = *(_QWORD *)(v49 + 56);
              if ( *(_BYTE *)(v50 + 173) == 12 && *(_BYTE *)(v50 + 176) == 2 )
                goto LABEL_93;
            }
          }
        }
      }
    }
    if ( *(_BYTE *)(v36 + 24) == 1 && (unsigned __int8)(*(_BYTE *)(v36 + 56) - 100) <= 1u )
      v36 = *(_QWORD *)(*(_QWORD *)(v36 + 72) + 16LL);
LABEL_69:
    v16 = (_DWORD *)v63;
    if ( !(unsigned int)sub_717510(v36, v63, 1) )
    {
      sub_724E30(&v63);
      v22 = qword_4D03C50;
LABEL_17:
      if ( (*(_DWORD *)(v22 + 16) & 0x40000100) == 0x40000100 )
      {
        if ( !dword_4F077C0
          || *(_BYTE *)(a1 + 16) != 1
          || (v16 = 0, !(unsigned int)sub_6EFD60(*(_QWORD *)(a1 + 144), 0, v22, dword_4F077C0, v18, v19)) )
        {
          if ( (unsigned int)sub_6E5430() )
            sub_6851C0(0x1Cu, v8);
          sub_6E6840(a1);
          goto LABEL_27;
        }
      }
      v22 = 1;
LABEL_19:
      v23 = 1;
      if ( v57 )
      {
LABEL_20:
        if ( a2 )
        {
          v24 = sub_73E250(v57);
          *(_QWORD *)(v24 + 28) = *(_QWORD *)(a1 + 68);
        }
        else if ( a6 || !(_BYTE)v22 )
        {
          if ( v60 )
            v39 = *(_QWORD *)&dword_4D03B80;
          else
            v39 = sub_731400(v57);
          if ( v11 )
          {
            v40 = sub_726700(23);
            *(_BYTE *)(v40 + 56) = 63;
            v24 = v40;
            *(_QWORD *)v40 = v39;
            *(_QWORD *)(v40 + 64) = v57;
          }
          else
          {
            v24 = sub_73DBF0(0, v39, v57);
          }
          if ( a6 )
            *(_QWORD *)(v24 + 28) = *a6;
          else
            *(_BYTE *)(v24 + 27) |= 2u;
        }
        else
        {
          v24 = sub_73E1B0(v57, v16);
        }
        if ( v23 )
        {
          sub_6E70E0((__int64 *)v24, a1);
          if ( v60 )
            sub_6F4B70((__m128i *)a1, a1, v25, v26, v27, v28);
        }
        else
        {
          *(_QWORD *)(a1 + 288) = v24;
        }
        goto LABEL_27;
      }
LABEL_81:
      v16 = 0;
      v62 = v22;
      v45 = sub_6F6F40((const __m128i *)a1, 0, v22, (__int64)v20, v18, v19);
      LOBYTE(v22) = v62;
      v57 = v45;
      goto LABEL_20;
    }
    goto LABEL_70;
  }
LABEL_9:
  *(_WORD *)(a1 + 18) &= 0xEFD7u;
}
