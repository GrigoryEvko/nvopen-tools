// Function: sub_11EBE80
// Address: 0x11ebe80
//
unsigned __int64 __fastcall sub_11EBE80(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  char *v7; // r14
  size_t v8; // rdx
  __int64 *v9; // rsi
  __int64 v10; // r8
  __int64 v11; // r9
  int v12; // edx
  char v13; // al
  char v14; // cl
  __int64 v15; // rbx
  __int64 v16; // rax
  unsigned __int64 v17; // r8
  int v19; // eax
  int v20; // eax
  char v21; // al
  _QWORD *v22; // r11
  char v23; // al
  __int64 v24; // rsi
  int v25; // eax
  __int64 v26; // rax
  __m128i v27; // rdi
  __int64 v28; // rdx
  __int64 v29; // rcx
  char v30; // al
  char v31; // al
  __int64 v32; // rdi
  unsigned __int8 *v33; // r10
  __int64 v34; // rax
  _BYTE *v35; // rdx
  _BYTE *v36; // rsi
  __int64 v37; // rcx
  __int64 v38; // rax
  _BYTE *v39; // r14
  bool v40; // zf
  unsigned __int64 v41; // rax
  __int64 *v42; // [rsp+8h] [rbp-F8h]
  __int16 v43; // [rsp+24h] [rbp-DCh]
  char v44; // [rsp+27h] [rbp-D9h]
  __int64 v45; // [rsp+28h] [rbp-D8h]
  int v46; // [rsp+30h] [rbp-D0h]
  int v47; // [rsp+38h] [rbp-C8h]
  int v48; // [rsp+3Ch] [rbp-C4h]
  int v49; // [rsp+40h] [rbp-C0h]
  int v50; // [rsp+44h] [rbp-BCh]
  __int64 v51; // [rsp+48h] [rbp-B8h]
  __int64 v52; // [rsp+48h] [rbp-B8h]
  int v53; // [rsp+50h] [rbp-B0h]
  unsigned int v54; // [rsp+54h] [rbp-ACh]
  size_t v55; // [rsp+58h] [rbp-A8h]
  __int64 v56; // [rsp+58h] [rbp-A8h]
  __int64 v57; // [rsp+58h] [rbp-A8h]
  int v58; // [rsp+68h] [rbp-98h] BYREF
  unsigned int v59; // [rsp+6Ch] [rbp-94h] BYREF
  __int64 v60; // [rsp+70h] [rbp-90h] BYREF
  __int64 v61; // [rsp+78h] [rbp-88h] BYREF
  __m128i v62; // [rsp+80h] [rbp-80h] BYREF
  __int64 v63; // [rsp+90h] [rbp-70h]
  __int64 v64; // [rsp+98h] [rbp-68h]
  __int64 v65; // [rsp+A0h] [rbp-60h]
  __int64 v66; // [rsp+A8h] [rbp-58h]
  __int64 v67; // [rsp+B0h] [rbp-50h]
  __int64 v68; // [rsp+B8h] [rbp-48h]
  __int16 v69; // [rsp+C0h] [rbp-40h]

  v3 = *(_QWORD *)(a2 - 32);
  if ( !v3 || *(_BYTE *)v3 || *(_QWORD *)(v3 + 24) != *(_QWORD *)(a2 + 80) )
  {
    sub_BD5D20(0);
    BUG();
  }
  v7 = (char *)sub_BD5D20(*(_QWORD *)(a2 - 32));
  v55 = v8;
  v54 = *(_DWORD *)(v3 + 36);
  v9 = (__int64 *)sub_B43CA0(a2);
  v51 = *(_QWORD *)(a2 + 8);
  if ( !*(_BYTE *)(a1 + 80)
    || !(unsigned __int8)sub_11E9B60(a1, v9, (__int64)v7, v55, v10, v11)
    || (v17 = sub_11DB650(a2, a3, 0, *(__int64 **)(a1 + 24), 1)) == 0 )
  {
    if ( !(unsigned __int8)sub_980AF0(**(_QWORD **)(a1 + 24), v7, v55, &v58) )
    {
      if ( v54 - 218 > 2 )
        return 0;
      v12 = *(unsigned __int8 *)(v51 + 8);
      v13 = *(_BYTE *)(v51 + 8);
      if ( (unsigned int)(v12 - 17) > 1 )
      {
        if ( (_BYTE)v12 != 2 )
        {
LABEL_26:
          if ( v12 != 17 )
          {
LABEL_12:
            if ( v13 == 3 )
            {
              v53 = 386;
              v48 = 228;
              v49 = 231;
              v50 = 227;
              goto LABEL_14;
            }
            return 0;
          }
          v14 = *(_BYTE *)(**(_QWORD **)(v51 + 16) + 8LL);
LABEL_11:
          v13 = v14;
          goto LABEL_12;
        }
      }
      else
      {
        v14 = *(_BYTE *)(**(_QWORD **)(v51 + 16) + 8LL);
        if ( v14 != 2 )
        {
          if ( v12 == 18 )
            goto LABEL_11;
          goto LABEL_26;
        }
      }
      v53 = 387;
      v48 = 229;
      v49 = 232;
      v50 = 234;
      goto LABEL_14;
    }
    switch ( v58 )
    {
      case 333:
        v53 = 386;
        v48 = 228;
        v49 = 231;
        v50 = 227;
        v54 = 218;
        break;
      case 334:
        v53 = 386;
        v48 = 228;
        v49 = 231;
        v50 = 227;
        v54 = 219;
        break;
      case 335:
        v53 = 387;
        v48 = 229;
        v49 = 232;
        v50 = 234;
        v54 = 219;
        break;
      case 336:
        v53 = 388;
        v48 = 230;
        v49 = 233;
        v50 = 235;
        v54 = 219;
        break;
      case 340:
        v53 = 386;
        v48 = 228;
        v49 = 231;
        v50 = 227;
        v54 = 220;
        break;
      case 341:
        v53 = 387;
        v48 = 229;
        v49 = 232;
        v50 = 234;
        v54 = 220;
        break;
      case 342:
        v53 = 388;
        v48 = 230;
        v49 = 233;
        v50 = 235;
        v54 = 220;
        break;
      case 349:
        v53 = 387;
        v48 = 229;
        v49 = 232;
        v50 = 234;
        v54 = 218;
        break;
      case 350:
        v53 = 388;
        v48 = 230;
        v49 = 233;
        v50 = 235;
        v54 = 218;
        break;
      default:
        return 0;
    }
    if ( !sub_B451C0(a2) || !sub_B451D0(a2) )
    {
      v26 = *(_QWORD *)(a1 + 40);
      v27.m128i_i64[1] = *(_QWORD *)(a1 + 24);
      v63 = 0;
      v28 = *(_QWORD *)(a1 + 48);
      v29 = *(_QWORD *)(a1 + 32);
      v66 = a2;
      v67 = v26;
      LODWORD(v26) = *(_DWORD *)(a2 + 4);
      v27.m128i_i64[0] = *(_QWORD *)(a1 + 16);
      v64 = v29;
      v65 = v28;
      v69 = 257;
      v62 = v27;
      v68 = 0;
      v61 = sub_9B4030(*(__int64 **)(a2 - 32 * (v26 & 0x7FFFFFF)), 156, 0, &v62);
      if ( (v61 & 0x1C) != 0 || !sub_989140(&v61, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 72LL), v51) )
      {
LABEL_14:
        v15 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
        if ( *(_BYTE *)v15 != 85 )
          return 0;
        if ( !sub_B45190(a2) )
          return 0;
        if ( !sub_B45190(v15) )
          return 0;
        v16 = *(_QWORD *)(v15 + 16);
        if ( !v16 || *(_QWORD *)(v16 + 8) )
          return 0;
        v19 = *(_DWORD *)(a3 + 104);
        *(_DWORD *)(a3 + 104) = -1;
        v47 = v19;
        v45 = *(_QWORD *)(a3 + 96);
        v44 = *(_BYTE *)(a3 + 110);
        v43 = *(_WORD *)(a3 + 108);
        v20 = sub_B49240(v15);
        v59 = 524;
        v46 = v20;
        v42 = *(__int64 **)(a1 + 24);
        v21 = sub_A73ED0((_QWORD *)(v15 + 72), 23);
        v22 = (_QWORD *)(v15 + 72);
        if ( !v21 && (v23 = sub_B49560(v15, 23), v22 = (_QWORD *)(v15 + 72), v17 = 0, !v23)
          || (v30 = sub_A73ED0(v22, 4), v17 = 0, v30)
          || (v31 = sub_B49560(v15, 4), v17 = 0, v31) )
        {
          v24 = *(_QWORD *)(v15 - 32);
          if ( v24 )
          {
            if ( !*(_BYTE *)v24 && *(_QWORD *)(v24 + 24) == *(_QWORD *)(v15 + 80) )
            {
              sub_981210(*v42, v24, &v59);
              v17 = 0;
            }
          }
        }
        v60 = 0;
        if ( v59 == v53 || (unsigned int)(v46 - 284) <= 1 )
        {
          if ( sub_B49E00(a2) )
          {
            BYTE4(v61) = 0;
            v62.m128i_i64[0] = (__int64)"log";
            LOWORD(v65) = 259;
            v38 = sub_B33BC0(a3, v54, *(_QWORD *)(v15 - 32LL * (*(_DWORD *)(v15 + 4) & 0x7FFFFFF)), v61, (__int64)&v62);
          }
          else
          {
            v38 = sub_11CC9B0(
                    *(_QWORD *)(v15 - 32LL * (*(_DWORD *)(v15 + 4) & 0x7FFFFFF)),
                    *(__int64 **)(a1 + 24),
                    v7,
                    v55,
                    a3,
                    &v60);
          }
          v39 = (_BYTE *)v38;
          v36 = *(_BYTE **)(v15 + 32 * (1LL - (*(_DWORD *)(v15 + 4) & 0x7FFFFFF)));
          if ( v46 == 285 )
          {
            v40 = *(_BYTE *)(a3 + 108) == 0;
            v62.m128i_i64[0] = (__int64)"cast";
            LOWORD(v65) = 259;
            if ( v40 )
            {
              v41 = sub_11DB4B0((__int64 *)a3, 0x2Cu, (unsigned __int64)v36, (__int64 **)v51, (__int64)&v62, 0, v61, 0);
            }
            else
            {
              BYTE4(v61) = 0;
              v41 = sub_B358C0(a3, 0x88u, (__int64)v36, v51, v61, (__int64)&v62, 0, 0, 0);
            }
            v36 = (_BYTE *)v41;
          }
          v35 = v39;
          HIDWORD(v61) = 0;
          v37 = (unsigned int)v61;
          v62.m128i_i64[0] = (__int64)"mul";
          LOWORD(v65) = 259;
        }
        else
        {
          if ( v59 == v50 )
            goto LABEL_74;
          if ( v59 != v48 && v59 != v49 && (v46 & 0xFFFFFFFD) != 0x58 )
            goto LABEL_40;
          if ( v46 == 88 )
          {
LABEL_74:
            v33 = sub_AD8DD0(*(_QWORD *)(a2 + 8), 2.718281828459045);
          }
          else
          {
            v32 = *(_QWORD *)(a2 + 8);
            if ( v46 == 90 || v59 == v49 )
              v33 = sub_AD8DD0(v32, 2.0);
            else
              v33 = sub_AD8DD0(v32, 10.0);
          }
          v52 = (__int64)v33;
          if ( sub_B49E00(a2) )
          {
            BYTE4(v61) = 0;
            v62.m128i_i64[0] = (__int64)"log";
            LOWORD(v65) = 259;
            v34 = sub_B33BC0(a3, v54, v52, v61, (__int64)&v62);
          }
          else
          {
            v34 = sub_11CC9B0(v52, *(__int64 **)(a1 + 24), v7, v55, a3, &v60);
          }
          v35 = (_BYTE *)v34;
          v62.m128i_i64[0] = (__int64)"mul";
          LOWORD(v65) = 259;
          v36 = *(_BYTE **)(v15 - 32LL * (*(_DWORD *)(v15 + 4) & 0x7FFFFFF));
          HIDWORD(v61) = 0;
          v37 = (unsigned int)v61;
        }
        v57 = sub_A826E0((unsigned int **)a3, v36, v35, v37, (__int64)&v62, 0);
        sub_11EA700(a1);
        sub_11EADF0(a1);
        v17 = v57;
LABEL_40:
        *(_QWORD *)(a3 + 96) = v45;
        *(_DWORD *)(a3 + 104) = v47;
        *(_WORD *)(a3 + 108) = v43;
        *(_BYTE *)(a3 + 110) = v44;
        return v17;
      }
    }
    LOWORD(v65) = 257;
    v25 = sub_B45210(a2);
    BYTE4(v61) = 1;
    LODWORD(v61) = v25;
    v56 = sub_B33BC0(a3, v54, *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), v61, (__int64)&v62);
    sub_B47C00(v56, a2, 0, 0);
    v17 = v56;
    if ( !v56 )
      return 0;
    if ( *(_BYTE *)v56 == 85 )
      *(_WORD *)(v56 + 2) = *(_WORD *)(v56 + 2) & 0xFFFC | *(_WORD *)(a2 + 2) & 3;
  }
  return v17;
}
