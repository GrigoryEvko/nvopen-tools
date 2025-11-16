// Function: sub_202E5A0
// Address: 0x202e5a0
//
__int64 __fastcall sub_202E5A0(__int64 **a1, unsigned __int64 a2, unsigned int a3, __m128i a4, __m128i a5, __m128i a6)
{
  unsigned int *v8; // rax
  __int64 v9; // rax
  unsigned int v10; // eax
  __int64 v11; // r8
  __int64 v12; // r9
  unsigned int v13; // r13d
  __int64 v15; // rcx
  const __m128i *v16; // r9
  unsigned int v17; // edx
  unsigned __int64 v18; // r8
  unsigned int v19; // edx
  unsigned int v20; // edx
  __int64 v21; // rax
  char v22; // di
  __int64 v23; // rdx
  __int64 *v24; // rax
  __int64 v25; // rcx
  __int64 v26; // rax
  __int64 v27; // rax
  char v28; // r15
  __int64 v29; // rax
  unsigned int v30; // edx
  __int64 v31; // rax
  char v32; // di
  __int64 v33; // rdx
  __int64 *v34; // rax
  __int64 v35; // rcx
  __int64 v36; // rax
  __int64 v37; // rax
  char v38; // r15
  __int64 v39; // rax
  unsigned int v40; // edx
  unsigned int v41; // edx
  unsigned int v42; // edx
  unsigned int v43; // edx
  unsigned int v44; // edx
  unsigned int v45; // edx
  unsigned int v46; // edx
  unsigned int v47; // edx
  unsigned int v48; // edx
  unsigned int v49; // edx
  unsigned int v50; // edx
  unsigned int v51; // edx
  unsigned int v52; // edx
  unsigned int v53; // ebx
  unsigned int v54; // eax
  unsigned int v55; // edx
  unsigned int v56; // ebx
  unsigned int v57; // eax
  __int64 v58; // [rsp-8h] [rbp-1A0h]
  __int64 v59; // [rsp+0h] [rbp-198h]
  char v60[8]; // [rsp+148h] [rbp-50h] BYREF
  __int64 v61; // [rsp+150h] [rbp-48h]
  char v62[8]; // [rsp+158h] [rbp-40h] BYREF
  __int64 v63; // [rsp+160h] [rbp-38h]

  v8 = (unsigned int *)(*(_QWORD *)(a2 + 32) + 40LL * a3);
  v9 = *(_QWORD *)(*(_QWORD *)v8 + 40LL) + 16LL * v8[2];
  v10 = sub_2016240(a1, a2, *(_BYTE *)v9, *(_QWORD *)(v9 + 8), 0, 0, 0);
  if ( (_BYTE)v10 )
    return 0;
  v13 = v10;
  switch ( *(_WORD *)(a2 + 24) )
  {
    case 0x65:
      v15 = (__int64)sub_202E540((__int64)a1, a2, a4, *(double *)a5.m128i_i64, a6, v58, v59, v11, v12);
      v18 = v46;
      goto LABEL_7;
    case 0x67:
    case 0x80:
    case 0x81:
    case 0x82:
    case 0x8E:
    case 0x8F:
    case 0x90:
    case 0x9D:
    case 0xAF:
      v15 = (__int64)sub_202A670((__int64)a1, a2, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
      v18 = v19;
      goto LABEL_7;
    case 0x6A:
      v15 = (__int64)sub_202ACC0(a1, a2, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
      v18 = v44;
      goto LABEL_7;
    case 0x6B:
      v15 = (__int64)sub_202D4A0(a1, a2, a4, *(double *)a5.m128i_i64, a6);
      v18 = v45;
      goto LABEL_7;
    case 0x6D:
      v15 = (__int64)sub_202AAF0((__int64)a1, (_QWORD *)a2, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
      v18 = v40;
      goto LABEL_7;
    case 0x87:
      v15 = (__int64)sub_2029F50((__int64)a1, a2);
      v18 = v41;
      goto LABEL_7;
    case 0x89:
      v15 = sub_202DEF0((__int64)a1, a2, (__m128)a4, *(double *)a5.m128i_i64, a6);
      v18 = v42;
      goto LABEL_7;
    case 0x91:
      v15 = (__int64)sub_202D8A0((__int64)a1, a2, a4, *(double *)a5.m128i_i64, a6);
      v18 = v43;
      goto LABEL_7;
    case 0x92:
    case 0x93:
      v21 = *(_QWORD *)(a2 + 40);
      v22 = *(_BYTE *)v21;
      v23 = *(_QWORD *)(v21 + 8);
      v24 = *(__int64 **)(a2 + 32);
      v60[0] = v22;
      v25 = *v24;
      v26 = *((unsigned int *)v24 + 2);
      v61 = v23;
      v27 = *(_QWORD *)(v25 + 40) + 16 * v26;
      v28 = *(_BYTE *)v27;
      v29 = *(_QWORD *)(v27 + 8);
      v62[0] = v28;
      v63 = v29;
      if ( v28 == v22 )
      {
        if ( v28 || v29 == v23 )
          goto LABEL_14;
      }
      else if ( v22 )
      {
        v53 = sub_2021900(v22);
        goto LABEL_34;
      }
      v53 = sub_1F58D40((__int64)v60);
LABEL_34:
      if ( v28 )
        v54 = sub_2021900(v28);
      else
        v54 = sub_1F58D40((__int64)v62);
      if ( v54 > v53 )
        goto LABEL_37;
      goto LABEL_14;
    case 0x95:
    case 0x96:
    case 0x97:
      v15 = (__int64)sub_202B2D0((__int64)a1, a2, a4, a5, a6);
      v18 = v20;
      goto LABEL_7;
    case 0x98:
    case 0x99:
      v31 = *(_QWORD *)(a2 + 40);
      v32 = *(_BYTE *)v31;
      v33 = *(_QWORD *)(v31 + 8);
      v34 = *(__int64 **)(a2 + 32);
      v60[0] = v32;
      v35 = *v34;
      v36 = *((unsigned int *)v34 + 2);
      v61 = v33;
      v37 = *(_QWORD *)(v35 + 40) + 16 * v36;
      v38 = *(_BYTE *)v37;
      v39 = *(_QWORD *)(v37 + 8);
      v62[0] = v38;
      v63 = v39;
      if ( v32 == v38 )
      {
        if ( v38 || v39 == v33 )
        {
LABEL_14:
          v15 = (__int64)sub_202A670((__int64)a1, a2, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
          v18 = v30;
          goto LABEL_7;
        }
      }
      else if ( v32 )
      {
        v56 = sub_2021900(v32);
        goto LABEL_40;
      }
      v56 = sub_1F58D40((__int64)v60);
LABEL_40:
      if ( v38 )
        v57 = sub_2021900(v38);
      else
        v57 = sub_1F58D40((__int64)v62);
      if ( v57 <= v56 )
        goto LABEL_14;
LABEL_37:
      v15 = (__int64)sub_202D8A0((__int64)a1, a2, a4, *(double *)a5.m128i_i64, a6);
      v18 = v55;
LABEL_7:
      if ( !v15 )
        return 0;
      if ( a2 == v15 )
        return 1;
      else
        sub_2013400((__int64)a1, a2, 0, v15, (__m128i *)v18, v16);
      return v13;
    case 0x9A:
      v15 = (__int64)sub_202E250((__int64)a1, a2, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
      v18 = v52;
      goto LABEL_7;
    case 0x9E:
      v15 = sub_202A960((__int64)a1, a2, a4, *(double *)a5.m128i_i64, a6);
      v18 = v48;
      goto LABEL_7;
    case 0xBA:
      v15 = (__int64)sub_202CEF0((__int64)a1, a2);
      v18 = v49;
      goto LABEL_7;
    case 0xEC:
      v15 = (__int64)sub_202BDB0((__int64 *)a1, a2);
      v18 = v50;
      goto LABEL_7;
    case 0xED:
      v15 = sub_202B3A0((__int64)a1, a2);
      v18 = v51;
      goto LABEL_7;
    case 0xEE:
      v15 = (__int64)sub_202C610((__int64 *)a1, a2);
      v18 = v47;
      goto LABEL_7;
    case 0xF6:
    case 0xF7:
    case 0xF8:
    case 0xF9:
    case 0xFA:
    case 0xFB:
    case 0xFC:
    case 0xFD:
    case 0xFE:
    case 0xFF:
    case 0x100:
    case 0x101:
    case 0x102:
      v15 = sub_202A420((__int64)a1, a2, a3, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
      v18 = v17;
      goto LABEL_7;
    default:
      sub_16BD130("Do not know how to split this operator's operand!\n", 1u);
  }
}
