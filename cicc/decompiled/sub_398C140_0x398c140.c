// Function: sub_398C140
// Address: 0x398c140
//
__int64 __fastcall sub_398C140(__int64 a1, char a2, __int64 a3, __int64 a4, __int64 a5, __int64 **a6)
{
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // r8
  void (*v11)(); // rax
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // r14
  __int64 v15; // rdi
  __int64 v16; // r8
  void (*v17)(); // rax
  __int64 v18; // rdi
  void (*v19)(); // rax
  __int64 v20; // r14
  __int64 v21; // rdi
  void (*v22)(); // rax
  __int64 v23; // rdi
  int v24; // r15d
  unsigned __int16 v25; // ax
  int v26; // esi
  __int64 *v27; // rdi
  __int64 v28; // rax
  __int64 *v29; // rax
  __int64 v30; // rdx
  __int64 *v31; // r15
  __int64 v32; // rdi
  _QWORD *v33; // r12
  __int64 v34; // r9
  __int64 v35; // r14
  __int64 v36; // r13
  void (*v37)(); // r8
  __int64 v38; // rdi
  void (*v39)(); // rax
  __int64 v40; // rax
  __int64 *v41; // rdx
  __int64 *v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rdi
  __int64 v45; // r8
  void (*v46)(); // rax
  __int64 v48; // rsi
  int v49; // edi
  int v50; // r9d
  int v51; // r11d
  char *v52; // rax
  __int64 v53; // rdx
  int v54; // r9d
  char v55; // al
  __int64 v56; // rdx
  char **v57; // rdx
  char v58; // al
  __m128i *v59; // rsi
  char v60; // dl
  const char **v61; // rdi
  __m128i v62; // xmm0
  __int64 v63; // [rsp+0h] [rbp-150h]
  __int64 v64; // [rsp+18h] [rbp-138h]
  int v65; // [rsp+24h] [rbp-12Ch]
  __int64 v66; // [rsp+28h] [rbp-128h]
  void (*v67)(); // [rsp+30h] [rbp-120h]
  __int64 *v68; // [rsp+40h] [rbp-110h]
  int v69; // [rsp+48h] [rbp-108h]
  _QWORD v71[2]; // [rsp+50h] [rbp-100h] BYREF
  _QWORD v72[2]; // [rsp+60h] [rbp-F0h] BYREF
  _QWORD v73[2]; // [rsp+70h] [rbp-E0h] BYREF
  __m128i v74; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v75; // [rsp+90h] [rbp-C0h]
  char *v76; // [rsp+A0h] [rbp-B0h] BYREF
  char v77; // [rsp+B0h] [rbp-A0h]
  char v78; // [rsp+B1h] [rbp-9Fh]
  __m128i v79; // [rsp+C0h] [rbp-90h] BYREF
  __int64 v80; // [rsp+D0h] [rbp-80h]
  const char *v81; // [rsp+E0h] [rbp-70h] BYREF
  _QWORD *v82; // [rsp+E8h] [rbp-68h]
  __int16 v83; // [rsp+F0h] [rbp-60h]
  __m128i v84; // [rsp+100h] [rbp-50h] BYREF
  __int64 v85; // [rsp+110h] [rbp-40h]

  v8 = *(_QWORD *)(a5 + 616);
  v9 = *(_QWORD *)(a1 + 8);
  v71[0] = a3;
  v71[1] = a4;
  if ( v8 )
    a5 = v8;
  v64 = a5;
  v10 = *(_QWORD *)(v9 + 256);
  v11 = *(void (**)())(*(_QWORD *)v10 + 104LL);
  v81 = "Length of Public ";
  v84.m128i_i64[0] = (__int64)&v81;
  v84.m128i_i64[1] = (__int64)" Info";
  LOWORD(v85) = 770;
  v83 = 1283;
  v82 = v71;
  if ( v11 != nullsub_580 )
  {
    ((void (__fastcall *)(__int64, __m128i *, __int64))v11)(v10, &v84, 1);
    v9 = *(_QWORD *)(a1 + 8);
  }
  v83 = 1283;
  LOWORD(v85) = 770;
  v81 = "pub";
  v82 = v71;
  v84.m128i_i64[0] = (__int64)&v81;
  v84.m128i_i64[1] = (__int64)"_begin";
  v12 = sub_396F530(v9, (__int64)&v84);
  v13 = *(_QWORD *)(a1 + 8);
  v14 = v12;
  v81 = "pub";
  v83 = 1283;
  v84.m128i_i64[1] = (__int64)"_end";
  v82 = v71;
  v84.m128i_i64[0] = (__int64)&v81;
  LOWORD(v85) = 770;
  v63 = sub_396F530(v13, (__int64)&v84);
  sub_396F380(*(_QWORD *)(a1 + 8));
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 256LL) + 176LL))(
    *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL),
    v14,
    0);
  v15 = *(_QWORD *)(a1 + 8);
  v16 = *(_QWORD *)(v15 + 256);
  v17 = *(void (**)())(*(_QWORD *)v16 + 104LL);
  v84.m128i_i64[0] = (__int64)"DWARF Version";
  LOWORD(v85) = 259;
  if ( v17 != nullsub_580 )
  {
    ((void (__fastcall *)(__int64, __m128i *, __int64))v17)(v16, &v84, 1);
    v15 = *(_QWORD *)(a1 + 8);
  }
  sub_396F320(v15, 2);
  v18 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL);
  v19 = *(void (**)())(*(_QWORD *)v18 + 104LL);
  v84.m128i_i64[0] = (__int64)"Offset of Compilation Unit Info";
  LOWORD(v85) = 259;
  if ( v19 != nullsub_580 )
    ((void (__fastcall *)(__int64, __m128i *, __int64))v19)(v18, &v84, 1);
  sub_398A7D0(a1, v64);
  v20 = *(_QWORD *)(a1 + 8);
  v21 = *(_QWORD *)(v20 + 256);
  v22 = *(void (**)())(*(_QWORD *)v21 + 104LL);
  v84.m128i_i64[0] = (__int64)"Compilation Unit Length";
  LOWORD(v85) = 259;
  if ( v22 != nullsub_580 )
  {
    ((void (__fastcall *)(__int64, __m128i *, __int64))v22)(v21, &v84, 1);
    v20 = *(_QWORD *)(a1 + 8);
  }
  if ( (unsigned __int16)sub_398C0A0(*(_QWORD *)(v64 + 200)) > 4u )
  {
    v23 = *(_QWORD *)(v64 + 200);
    v24 = 8 * (*(_BYTE *)(v23 + 4513) != 0);
  }
  else
  {
    v23 = *(_QWORD *)(v64 + 200);
    v24 = 0;
  }
  v25 = sub_398C0A0(v23);
  sub_396F340(v20, (v25 > 4u) + v24 + *(_DWORD *)(v64 + 28) + 4 + 7);
  v26 = *((_DWORD *)a6 + 2);
  if ( v26 )
  {
    v27 = *a6;
    v28 = **a6;
    if ( v28 != -8 && v28 )
    {
      v31 = *a6;
    }
    else
    {
      v29 = v27 + 1;
      do
      {
        do
        {
          v30 = *v29;
          v31 = v29++;
        }
        while ( v30 == -8 );
      }
      while ( !v30 );
    }
    v68 = &v27[v26];
    if ( v68 != v31 )
    {
      while ( 1 )
      {
        v32 = *(_QWORD *)(a1 + 8);
        v33 = (_QWORD *)*v31;
        v34 = *(_QWORD *)(v32 + 256);
        v35 = *(_QWORD *)(*v31 + 8);
        v36 = *v31 + 16;
        v37 = *(void (**)())(*(_QWORD *)v34 + 104LL);
        v84.m128i_i64[0] = (__int64)"DIE offset";
        LOWORD(v85) = 259;
        if ( v37 != nullsub_580 )
        {
          ((void (__fastcall *)(__int64, __m128i *, __int64))v37)(v34, &v84, 1);
          v32 = *(_QWORD *)(a1 + 8);
        }
        sub_396F340(v32, *(_DWORD *)(v35 + 16));
        if ( !a2 )
          goto LABEL_23;
        if ( *(_WORD *)(v35 + 28) == 17 )
        {
LABEL_39:
          v50 = 16;
          v49 = 0;
          v51 = 1;
        }
        else
        {
          sub_3981F30((__int64)&v81, v35, 71);
          v48 = v35;
          if ( (_DWORD)v81 )
            v48 = (__int64)v82;
          sub_3981F30((__int64)&v84, v48, 63);
          v49 = v84.m128i_i32[0] == 0;
          switch ( *(_WORD *)(v35 + 28) )
          {
            case 2:
            case 4:
            case 0x13:
            case 0x17:
              v51 = 1;
              v49 = *(_WORD *)(*(_QWORD *)(v64 + 80) + 24LL) != 4;
              v50 = (v49 << 7) | 0x10;
              break;
            case 0x16:
            case 0x21:
            case 0x24:
              v50 = 144;
              v49 = 1;
              v51 = 1;
              break;
            case 0x28:
              v50 = 160;
              v49 = 1;
              v51 = 2;
              break;
            case 0x2E:
              v51 = 3;
              v50 = (v49 << 7) | 0x30;
              break;
            case 0x34:
              v51 = 2;
              v50 = (v49 << 7) | 0x20;
              break;
            case 0x39:
              goto LABEL_39;
            default:
              v50 = 0;
              v49 = 0;
              v51 = 0;
              break;
          }
        }
        v69 = v50;
        v65 = v51;
        v66 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL);
        v67 = *(void (**)())(*(_QWORD *)v66 + 104LL);
        v52 = sub_14E9930(v49);
        v78 = 1;
        v73[0] = v52;
        v73[1] = v53;
        v81 = (const char *)v73;
        v83 = 261;
        v76 = ", ";
        v77 = 3;
        v72[0] = sub_14E9890(v65);
        v54 = v69;
        v74.m128i_i64[0] = (__int64)"Kind: ";
        v74.m128i_i64[1] = (__int64)v72;
        v55 = v77;
        v72[1] = v56;
        LOWORD(v75) = 1283;
        if ( !v77 )
        {
          LOWORD(v80) = 256;
LABEL_52:
          LOWORD(v85) = 256;
LABEL_53:
          if ( v67 == nullsub_580 )
            goto LABEL_54;
          goto LABEL_50;
        }
        if ( v77 != 1 )
          break;
        v62 = _mm_loadu_si128(&v74);
        v80 = v75;
        v58 = v83;
        v79 = v62;
        if ( !(_BYTE)v83 )
          goto LABEL_52;
        if ( (_BYTE)v83 == 1 )
        {
LABEL_62:
          v84 = _mm_loadu_si128(&v79);
          v85 = v80;
          goto LABEL_53;
        }
        if ( BYTE1(v80) != 1 )
          goto LABEL_46;
        v59 = (__m128i *)v79.m128i_i64[0];
        v60 = 3;
LABEL_47:
        v61 = (const char **)v81;
        if ( HIBYTE(v83) != 1 )
        {
          v61 = &v81;
          v58 = 2;
        }
        v84.m128i_i64[0] = (__int64)v59;
        v84.m128i_i64[1] = (__int64)v61;
        LOBYTE(v85) = v60;
        BYTE1(v85) = v58;
        if ( v67 == nullsub_580 )
          goto LABEL_54;
LABEL_50:
        ((void (__fastcall *)(__int64, __m128i *, __int64))v67)(v66, &v84, 1);
        v54 = v69;
LABEL_54:
        sub_396F300(*(_QWORD *)(a1 + 8), v54);
LABEL_23:
        v38 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL);
        v39 = *(void (**)())(*(_QWORD *)v38 + 104LL);
        v84.m128i_i64[0] = (__int64)"External Name";
        LOWORD(v85) = 259;
        if ( v39 != nullsub_580 )
        {
          ((void (__fastcall *)(__int64, __m128i *, __int64))v39)(v38, &v84, 1);
          v38 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL);
        }
        (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v38 + 400LL))(v38, v36, *v33 + 1LL);
        v40 = v31[1];
        v41 = v31 + 1;
        if ( v40 && v40 != -8 )
        {
          ++v31;
          if ( v41 == v68 )
            goto LABEL_31;
        }
        else
        {
          v42 = v31 + 2;
          do
          {
            do
            {
              v43 = *v42;
              v31 = v42++;
            }
            while ( v43 == -8 );
          }
          while ( !v43 );
          if ( v31 == v68 )
            goto LABEL_31;
        }
      }
      v57 = (char **)v76;
      if ( v78 != 1 )
      {
        v57 = &v76;
        v55 = 2;
      }
      BYTE1(v80) = v55;
      v58 = v83;
      v79.m128i_i64[0] = (__int64)&v74;
      v79.m128i_i64[1] = (__int64)v57;
      LOBYTE(v80) = 2;
      if ( !(_BYTE)v83 )
        goto LABEL_52;
      if ( (_BYTE)v83 == 1 )
        goto LABEL_62;
LABEL_46:
      v59 = &v79;
      v60 = 2;
      goto LABEL_47;
    }
  }
LABEL_31:
  v44 = *(_QWORD *)(a1 + 8);
  v45 = *(_QWORD *)(v44 + 256);
  v46 = *(void (**)())(*(_QWORD *)v45 + 104LL);
  v84.m128i_i64[0] = (__int64)"End Mark";
  LOWORD(v85) = 259;
  if ( v46 != nullsub_580 )
  {
    ((void (__fastcall *)(__int64, __m128i *, __int64))v46)(v45, &v84, 1);
    v44 = *(_QWORD *)(a1 + 8);
  }
  sub_396F340(v44, 0);
  return (*(__int64 (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 256LL) + 176LL))(
           *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL),
           v63,
           0);
}
