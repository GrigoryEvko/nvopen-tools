// Function: sub_7E88E0
// Address: 0x7e88e0
//
__int64 __fastcall sub_7E88E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __m128i *v7; // r15
  __int64 result; // rax
  __int64 v9; // rdx
  __m128i *v10; // r13
  __int64 v11; // rcx
  char v12; // bl
  __int64 v13; // rsi
  _QWORD *v14; // rax
  __int64 v15; // rcx
  _QWORD *v16; // rbx
  int *v17; // rsi
  _QWORD *v18; // r12
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rcx
  _BYTE *v24; // rbx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 i; // rax
  int v28; // eax
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // rsi
  char v34; // r12
  char v35; // si
  char v36; // si
  _QWORD *v37; // rax
  __int64 v38; // rcx
  const __m128i *v39; // rdi
  _QWORD *v40; // rax
  char v41; // al
  char v42; // al
  _QWORD *v43; // rax
  __int64 *v44; // rdx
  _QWORD *v45; // rax
  _BOOL4 v46; // eax
  _BOOL4 v47; // eax
  __m128i *v48; // rax
  _QWORD *v49; // rax
  _QWORD *v50; // rax
  _QWORD *v51; // rax
  __int64 v52; // [rsp+8h] [rbp-88h]
  __int64 v53; // [rsp+8h] [rbp-88h]
  __int64 v54; // [rsp+10h] [rbp-80h]
  __m128i *v55; // [rsp+10h] [rbp-80h]
  __int64 v56; // [rsp+10h] [rbp-80h]
  __int64 v57; // [rsp+10h] [rbp-80h]
  __int64 v58; // [rsp+10h] [rbp-80h]
  __int64 v59; // [rsp+10h] [rbp-80h]
  __m128i *v60; // [rsp+10h] [rbp-80h]
  __int64 v61; // [rsp+18h] [rbp-78h]
  __int64 v62; // [rsp+18h] [rbp-78h]
  __int64 v63; // [rsp+18h] [rbp-78h]
  unsigned __int8 v64; // [rsp+18h] [rbp-78h]
  unsigned __int8 v65; // [rsp+18h] [rbp-78h]
  __int64 v66; // [rsp+18h] [rbp-78h]
  __int64 v67; // [rsp+20h] [rbp-70h]
  __m128i *v68; // [rsp+28h] [rbp-68h]
  __int64 v69; // [rsp+30h] [rbp-60h]
  bool v70; // [rsp+3Eh] [rbp-52h]
  char v71; // [rsp+3Fh] [rbp-51h]
  int v72[20]; // [rsp+40h] [rbp-50h] BYREF

  v7 = *(__m128i **)(a1 + 72);
  result = *(unsigned __int8 *)(a1 + 56);
  if ( v7[1].m128i_i8[8] == 1 && (v7[3].m128i_i8[10] & 1) != 0 )
  {
    v9 = v7[3].m128i_u8[8];
    if ( (_BYTE)v9 == 103 )
    {
      if ( (_BYTE)result != 91 )
      {
        v10 = (__m128i *)v7[1].m128i_i64[0];
        v67 = v7[4].m128i_i64[1];
        v11 = *(_QWORD *)(v67 + 16);
        v69 = *(_QWORD *)a1;
        v36 = *(_BYTE *)(a1 + 25);
        v71 = v36 & 1;
        v13 = (v36 & 4) != 0;
        v70 = v13;
        if ( !dword_4D04810 )
        {
LABEL_65:
          v68 = 0;
          v62 = *(_QWORD *)(v11 + 16);
LABEL_66:
          if ( v10 )
          {
            v59 = v11;
            if ( (unsigned int)sub_731D60((__int64)v10, v13, v9, v11, a5, a6) )
            {
              v53 = v59;
              v60 = (__m128i *)sub_7E88C0(v10);
              if ( v68 )
              {
                v68 = (__m128i *)sub_73DF90((__int64)v68, v10->m128i_i64);
                v16 = sub_730FF0((const __m128i *)a1);
                v51 = sub_730FF0((const __m128i *)a1);
              }
              else
              {
                v16 = sub_730FF0((const __m128i *)a1);
                v51 = sub_730FF0((const __m128i *)a1);
                v68 = v10;
              }
              v44 = (__int64 *)v60;
              v18 = v51;
              v16[9] = v53;
              *(_QWORD *)(v53 + 16) = v60;
              v51[9] = v62;
              if ( !v60 )
                goto LABEL_68;
              v10 = v60;
            }
            else
            {
              v16 = sub_730FF0((const __m128i *)a1);
              v18 = sub_730FF0((const __m128i *)a1);
              v16[9] = v59;
              *(_QWORD *)(v59 + 16) = v10;
              v18[9] = v62;
            }
            v44 = sub_73B8B0(v10, 0);
LABEL_68:
            *(_QWORD *)(v62 + 16) = v44;
            *(_QWORD *)(v67 + 16) = v16;
            v16[2] = v18;
LABEL_44:
            v17 = (int *)v7;
            sub_730620(a1, v7);
            goto LABEL_45;
          }
LABEL_67:
          v56 = v11;
          v16 = sub_730FF0((const __m128i *)a1);
          v43 = sub_730FF0((const __m128i *)a1);
          v44 = 0;
          v18 = v43;
          v16[9] = v56;
          *(_QWORD *)(v56 + 16) = 0;
          v43[9] = v62;
          goto LABEL_68;
        }
        v9 = 0;
        if ( !v10 )
        {
LABEL_80:
          v68 = 0;
          v62 = *(_QWORD *)(v11 + 16);
          goto LABEL_67;
        }
        goto LABEL_35;
      }
      return (__int64)sub_7E0A10(*(__m128i **)(a1 + 72));
    }
    if ( (_BYTE)v9 == 104 || (_BYTE)v9 == 91 )
    {
      if ( (_BYTE)result != 91 )
      {
        v10 = (__m128i *)v7[1].m128i_i64[0];
        v9 = (unsigned int)(v9 - 103);
        v67 = v7[4].m128i_i64[1];
        v11 = *(_QWORD *)(v67 + 16);
        v69 = *(_QWORD *)a1;
        v12 = *(_BYTE *)(a1 + 25);
        v71 = v12 & 1;
        v13 = dword_4D04810;
        v70 = (v12 & 4) != 0;
        if ( !dword_4D04810 )
        {
          if ( (_BYTE)v9 != 1 )
          {
LABEL_14:
            v61 = *(_QWORD *)(v67 + 16);
            v14 = sub_730FF0((const __m128i *)a1);
            v68 = 0;
            v15 = v61;
            v16 = v14;
LABEL_15:
            v16[9] = v15;
            *(_QWORD *)(v15 + 16) = v10;
            v10 = v7;
            *(_QWORD *)(v67 + 16) = v16;
LABEL_16:
            v17 = (int *)v10;
            v18 = 0;
            sub_730620(a1, v10);
LABEL_45:
            if ( !v71 )
            {
              *(_BYTE *)(a1 + 58) &= ~1u;
              *(_BYTE *)(a1 + 25) &= ~1u;
            }
            if ( v68 )
            {
              sub_7E1780(a1, (__int64)v72);
              v17 = v72;
              sub_7E25D0((__int64)v68, v72);
            }
            v41 = *((_BYTE *)v16 + 56);
            if ( !v41 || v41 == 21 )
              sub_7313A0(v16[9], (__int64)v17, v19, v20, v21, v22);
            if ( *((_BYTE *)v16 + 24) == 1 )
            {
              sub_7DFF80((__int64)v16, (__int64)v17, v19, v20, v21, v22);
              if ( *((_BYTE *)v16 + 24) == 1 )
                sub_7E88E0(v16);
            }
            if ( v18 )
            {
              v42 = *((_BYTE *)v18 + 56);
              if ( !v42 || v42 == 21 )
                sub_7313A0(v18[9], (__int64)v17, v19, v20, v21, v22);
              if ( *((_BYTE *)v18 + 24) == 1 )
              {
                sub_7DFF80((__int64)v18, (__int64)v17, v19, v20, v21, v22);
                if ( *((_BYTE *)v18 + 24) == 1 )
                  sub_7E88E0(v18);
              }
            }
            result = v69;
            *(_QWORD *)a1 = v69;
            if ( v70 )
              return sub_7304E0(a1);
            return result;
          }
          goto LABEL_65;
        }
        if ( !v10 )
        {
          if ( (unsigned __int8)v9 > 1u )
            goto LABEL_14;
          goto LABEL_80;
        }
LABEL_35:
        v34 = 1;
        goto LABEL_36;
      }
      return (__int64)sub_7E0A10(*(__m128i **)(a1 + 72));
    }
  }
  if ( (_BYTE)result != 86 )
  {
    if ( (*(_BYTE *)(a1 + 58) & 1) == 0 || (unsigned __int8)(result - 103) <= 1u || (_BYTE)result == 91 )
      return result;
    v10 = (__m128i *)v7[1].m128i_i64[0];
    goto LABEL_19;
  }
  v10 = (__m128i *)v7[1].m128i_i64[0];
  if ( v10[1].m128i_i8[8] == 1 && (v10[3].m128i_i8[10] & 1) != 0 )
  {
    result = v10[3].m128i_u8[8];
    if ( (_BYTE)result == 103 )
    {
      v9 = 0;
LABEL_29:
      v34 = 0;
      v67 = v10[4].m128i_i64[1];
      v11 = *(_QWORD *)(v67 + 16);
      v69 = *(_QWORD *)a1;
      v35 = *(_BYTE *)(a1 + 25);
      v71 = v35 & 1;
      v13 = (v35 & 4) != 0;
      v70 = v13;
      if ( !dword_4D04810 )
      {
        if ( (unsigned __int8)v9 <= 1u )
        {
          v68 = 0;
          v62 = *(_QWORD *)(v11 + 16);
          goto LABEL_39;
        }
        v66 = *(_QWORD *)(v67 + 16);
        v49 = sub_730FF0((const __m128i *)a1);
        v68 = 0;
        v15 = v66;
        v16 = v49;
LABEL_73:
        v16[9] = v7;
        v7[1].m128i_i64[0] = v15;
        *(_QWORD *)(v67 + 16) = v16;
        goto LABEL_16;
      }
LABEL_36:
      v68 = 0;
      if ( (*(_BYTE *)(a1 + 60) & 2) != 0 )
      {
        v13 = (__int64)v10;
        v57 = v11;
        v64 = v9;
        v46 = sub_7E6EE0((__int64)v7, (__int64)v10);
        LOBYTE(v9) = v64;
        v11 = v57;
        if ( v46 || (v13 = (__int64)v7, v47 = sub_7E6EE0((__int64)v10, (__int64)v7), v9 = v64, v11 = v57, v47) )
        {
          v58 = v11;
          v65 = v9;
          v48 = (__m128i *)sub_7E88C0(v10);
          v68 = v10;
          v9 = v65;
          v11 = v58;
          v10 = v48;
        }
      }
      if ( (unsigned __int8)v9 <= 1u )
      {
        v62 = *(_QWORD *)(v11 + 16);
        if ( v34 )
          goto LABEL_66;
LABEL_39:
        v54 = v11;
        if ( (unsigned int)sub_731D60((__int64)v7, v13, v9, v11, a5, a6) )
        {
          v52 = v54;
          v55 = (__m128i *)sub_7E88C0(v7);
          if ( v68 )
          {
            v68 = (__m128i *)sub_73DF90((__int64)v68, v7->m128i_i64);
            v16 = sub_730FF0((const __m128i *)a1);
            v37 = sub_730FF0((const __m128i *)a1);
          }
          else
          {
            v16 = sub_730FF0((const __m128i *)a1);
            v37 = sub_730FF0((const __m128i *)a1);
            v68 = v7;
          }
          v38 = v52;
          v18 = v37;
          v7 = v55;
        }
        else
        {
          v16 = sub_730FF0((const __m128i *)a1);
          v50 = sub_730FF0((const __m128i *)a1);
          v38 = v54;
          v18 = v50;
        }
        v16[9] = v7;
        v39 = v7;
        v7[1].m128i_i64[0] = v38;
        v7 = v10;
        *(_QWORD *)(v38 + 16) = 0;
        v40 = sub_73B8B0(v39, 0);
        v18[9] = v40;
        v40[2] = v62;
        *(_QWORD *)(v67 + 16) = v16;
        v16[2] = v18;
        goto LABEL_44;
      }
      v63 = v11;
      v45 = sub_730FF0((const __m128i *)a1);
      v15 = v63;
      v16 = v45;
      if ( v34 )
        goto LABEL_15;
      goto LABEL_73;
    }
    if ( (_BYTE)result == 91 )
    {
      v9 = 4294967284LL;
      goto LABEL_29;
    }
  }
  if ( (*(_BYTE *)(a1 + 58) & 1) == 0 )
    return result;
LABEL_19:
  v24 = sub_730FF0((const __m128i *)a1);
  for ( i = *(_QWORD *)a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v24[58] &= ~1u;
  v24[25] &= ~1u;
  *(_QWORD *)v24 = i;
  v28 = sub_731920((__int64)v7, 1, 0, v23, v25, v26);
  if ( v10 && v28 )
    v33 = (unsigned int)sub_731770((__int64)v10, 0, v29, v30, v31, v32);
  else
    v33 = v28 == 0;
  *((_QWORD *)v24 + 2) = sub_7E25B0((__int64)v7, v33, v29, v30, v31, v32);
  return sub_73D8E0(a1, 0x5Bu, *(_QWORD *)a1, *(_BYTE *)(a1 + 25) & 1, (__int64)v24);
}
