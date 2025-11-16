// Function: sub_1618840
// Address: 0x1618840
//
__int64 __fastcall sub_1618840(__int64 *a1, __int64 a2)
{
  __int64 v3; // r14
  __int64 *v4; // rbx
  __int64 result; // rax
  __int64 v6; // r13
  __int64 v7; // r12
  __int64 v8; // rdi
  __int64 v9; // rdx
  __int64 *v10; // r12
  __int64 v11; // rax
  __int64 v12; // rax
  size_t v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r9
  __int64 v16; // r8
  const char *v17; // rsi
  __m128i *v18; // rdi
  unsigned __int64 v19; // rax
  __m128i si128; // xmm0
  __int64 v21; // rdi
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdi
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdi
  char **v28; // rbx
  __int64 v29; // r14
  _BYTE *v30; // rax
  __int64 v31; // rax
  size_t v32; // rdx
  _BYTE *v33; // rdi
  const char *v34; // rsi
  _BYTE *v35; // rax
  char *v36; // rsi
  __int64 v37; // rdi
  __int64 v38; // rdx
  __int64 v39; // r13
  __int64 v40; // rcx
  __int64 v41; // rdi
  __int64 v42; // r8
  __int64 v43; // r9
  _BYTE *v44; // rax
  __m128i *v45; // rdx
  __int64 v46; // rax
  _BYTE *v47; // rax
  __int64 v48; // rax
  _WORD *v49; // rdx
  __int64 v50; // rdi
  __int64 v51; // rdx
  __m128i v52; // xmm0
  char *v53; // rsi
  _BYTE *v54; // rax
  __int64 v55; // rax
  __int64 v56; // r8
  __int64 v57; // r9
  _WORD *v58; // rdx
  __int64 v59; // rdi
  __int64 v60; // rcx
  __int64 v61; // rdx
  __m128i v62; // xmm0
  __m128i v63; // xmm0
  _BYTE *v64; // rax
  _BYTE *v65; // rdx
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // rax
  __m128i *v70; // rdx
  __int64 v71; // [rsp+0h] [rbp-70h]
  __int64 *v72; // [rsp+8h] [rbp-68h]
  __int64 v73; // [rsp+10h] [rbp-60h]
  size_t v74; // [rsp+18h] [rbp-58h]
  __int64 v75; // [rsp+18h] [rbp-58h]
  __int64 v76; // [rsp+20h] [rbp-50h]
  __int64 v77; // [rsp+20h] [rbp-50h]
  size_t v78; // [rsp+20h] [rbp-50h]
  __int64 v79; // [rsp+28h] [rbp-48h]
  char v80; // [rsp+37h] [rbp-39h]
  __int64 v81; // [rsp+38h] [rbp-38h]
  int v82; // [rsp+38h] [rbp-38h]
  int v83; // [rsp+38h] [rbp-38h]

  v3 = a2;
  v79 = sub_16135E0(*a1, a2);
  do
  {
    v4 = *(__int64 **)v79;
    result = *(_QWORD *)v79 + 8LL * *(unsigned int *)(v79 + 8);
    if ( result == *(_QWORD *)v79 )
      return result;
    v80 = 0;
    v6 = *(_QWORD *)v79 + 8LL * *(unsigned int *)(v79 + 8);
    do
    {
      while ( 1 )
      {
        v7 = *v4;
        result = sub_160EA80(*a1, *v4);
        if ( !result )
          break;
LABEL_4:
        if ( (__int64 *)v6 == ++v4 )
          goto LABEL_10;
      }
      v8 = *a1;
      v81 = sub_1614F20(*a1, v7);
      if ( v81 )
        goto LABEL_7;
      v11 = sub_16BA580(v8, v7, v9);
      v76 = sub_1263B40(v11, "Pass '");
      v12 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v3 + 16LL))(v3);
      v16 = v76;
      v17 = (const char *)v12;
      v18 = *(__m128i **)(v76 + 24);
      v19 = *(_QWORD *)(v76 + 16) - (_QWORD)v18;
      if ( v13 > v19 )
      {
        v68 = sub_16E7EE0(v76, v17);
        v18 = *(__m128i **)(v68 + 24);
        v16 = v68;
        if ( *(_QWORD *)(v68 + 16) - (_QWORD)v18 <= 0x14u )
          goto LABEL_60;
      }
      else
      {
        if ( v13 )
        {
          v75 = v76;
          v78 = v13;
          memcpy(v18, v17, v13);
          v16 = v75;
          v69 = *(_QWORD *)(v75 + 16);
          v70 = (__m128i *)(*(_QWORD *)(v75 + 24) + v78);
          *(_QWORD *)(v75 + 24) = v70;
          v18 = v70;
          v19 = v69 - (_QWORD)v70;
        }
        if ( v19 <= 0x14 )
        {
LABEL_60:
          v16 = sub_16E7EE0(v16, "' is not initialized.", 21, v14, v16, v15, v71, v72, v73);
          goto LABEL_19;
        }
      }
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F55310);
      v18[1].m128i_i32[0] = 1684372073;
      v18[1].m128i_i8[4] = 46;
      *v18 = si128;
      *(_QWORD *)(v16 + 24) += 21LL;
LABEL_19:
      v21 = v16;
      sub_1263B40(v16, "\n");
      v23 = sub_16BA580(v21, "\n", v22);
      v24 = sub_1263B40(v23, "Verify if there is a pass dependency cycle.");
      sub_1263B40(v24, "\n");
      v26 = sub_16BA580(v24, "\n", v25);
      v27 = sub_1263B40(v26, "Required Passes:");
      sub_1263B40(v27, "\n");
      v77 = *(_QWORD *)v79 + 8LL * *(unsigned int *)(v79 + 8);
      if ( v77 == *(_QWORD *)v79 )
        goto LABEL_7;
      v72 = v4;
      v28 = *(char ***)v79;
      v73 = v6;
      v71 = v3;
      while ( 1 )
      {
        v36 = *v28;
        if ( (char *)v7 == *v28 )
          break;
        v37 = *a1;
        v39 = sub_160EA80(*a1, (__int64)v36);
        if ( v39 )
        {
          v29 = sub_16BA580(v37, v36, v38);
          v30 = *(_BYTE **)(v29 + 24);
          if ( *(_BYTE **)(v29 + 16) == v30 )
          {
            v29 = sub_16E7EE0(v29, "\t", 1);
          }
          else
          {
            *v30 = 9;
            ++*(_QWORD *)(v29 + 24);
          }
          v31 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v39 + 16LL))(v39);
          v33 = *(_BYTE **)(v29 + 24);
          v34 = (const char *)v31;
          v35 = *(_BYTE **)(v29 + 16);
          if ( v35 - v33 < v32 )
          {
            v29 = sub_16E7EE0(v29, v34);
            v35 = *(_BYTE **)(v29 + 16);
            v33 = *(_BYTE **)(v29 + 24);
          }
          else if ( v32 )
          {
            v74 = v32;
            memcpy(v33, v34, v32);
            v65 = (_BYTE *)(*(_QWORD *)(v29 + 24) + v74);
            *(_QWORD *)(v29 + 24) = v65;
            v35 = *(_BYTE **)(v29 + 16);
            v33 = v65;
          }
          if ( v33 == v35 )
          {
            sub_16E7EE0(v29, "\n", 1);
          }
          else
          {
            *v33 = 10;
            ++*(_QWORD *)(v29 + 24);
          }
          goto LABEL_28;
        }
        v41 = sub_16BA580(v37, v36, v38);
        v44 = *(_BYTE **)(v41 + 24);
        if ( *(_BYTE **)(v41 + 16) == v44 )
        {
          v36 = "\t";
          v41 = sub_16E7EE0(v41, "\t", 1);
          v45 = *(__m128i **)(v41 + 24);
          if ( *(_QWORD *)(v41 + 16) - (_QWORD)v45 > 0x2Fu )
          {
LABEL_33:
            *v45 = _mm_load_si128((const __m128i *)&xmmword_3F55350);
            v45[1] = _mm_load_si128((const __m128i *)&xmmword_3F55360);
            v45[2] = _mm_load_si128((const __m128i *)&xmmword_3F55370);
            v47 = (_BYTE *)(*(_QWORD *)(v41 + 24) + 48LL);
            *(_QWORD *)(v41 + 24) = v47;
            if ( *(_BYTE **)(v41 + 16) != v47 )
              goto LABEL_34;
            goto LABEL_52;
          }
        }
        else
        {
          *v44 = 9;
          v45 = (__m128i *)(*(_QWORD *)(v41 + 24) + 1LL);
          v46 = *(_QWORD *)(v41 + 16);
          *(_QWORD *)(v41 + 24) = v45;
          if ( (unsigned __int64)(v46 - (_QWORD)v45) > 0x2F )
            goto LABEL_33;
        }
        v36 = "Error: Required pass not found! Possible causes:";
        v41 = sub_16E7EE0(v41, "Error: Required pass not found! Possible causes:", 48, v40, v42, v43, v71, v72, v73);
        v47 = *(_BYTE **)(v41 + 24);
        if ( *(_BYTE **)(v41 + 16) != v47 )
        {
LABEL_34:
          *v47 = 10;
          ++*(_QWORD *)(v41 + 24);
          goto LABEL_35;
        }
LABEL_52:
        v36 = "\n";
        sub_16E7EE0(v41, "\n", 1, v40, v42, v43, v71, v72, v73);
LABEL_35:
        v48 = sub_16BA580(v41, v36, v45);
        v49 = *(_WORD **)(v48 + 24);
        v50 = v48;
        if ( *(_QWORD *)(v48 + 16) - (_QWORD)v49 <= 1u )
        {
          v67 = sub_16E7EE0(v48, "\t\t", 2);
          v51 = *(_QWORD *)(v67 + 24);
          v50 = v67;
        }
        else
        {
          *v49 = 2313;
          v51 = *(_QWORD *)(v48 + 24) + 2LL;
          *(_QWORD *)(v48 + 24) = v51;
        }
        if ( (unsigned __int64)(*(_QWORD *)(v50 + 16) - v51) <= 0x2D )
        {
          v53 = "- Pass misconfiguration (e.g.: missing macros)";
          v50 = sub_16E7EE0(v50, "- Pass misconfiguration (e.g.: missing macros)", 46);
          v54 = *(_BYTE **)(v50 + 24);
        }
        else
        {
          v52 = _mm_load_si128((const __m128i *)&xmmword_3F55380);
          v53 = (char *)10611;
          qmemcpy((void *)(v51 + 32), "issing macros)", 14);
          *(__m128i *)v51 = v52;
          *(__m128i *)(v51 + 16) = _mm_load_si128((const __m128i *)&xmmword_3F55390);
          v54 = (_BYTE *)(*(_QWORD *)(v50 + 24) + 46LL);
          *(_QWORD *)(v50 + 24) = v54;
        }
        if ( *(_BYTE **)(v50 + 16) == v54 )
        {
          v53 = "\n";
          sub_16E7EE0(v50, "\n", 1);
        }
        else
        {
          *v54 = 10;
          ++*(_QWORD *)(v50 + 24);
        }
        v55 = sub_16BA580(v50, v53, v51);
        v58 = *(_WORD **)(v55 + 24);
        v59 = v55;
        if ( *(_QWORD *)(v55 + 16) - (_QWORD)v58 <= 1u )
        {
          v66 = sub_16E7EE0(v55, "\t\t", 2);
          v61 = *(_QWORD *)(v66 + 24);
          v59 = v66;
        }
        else
        {
          v60 = 2313;
          *v58 = 2313;
          v61 = *(_QWORD *)(v55 + 24) + 2LL;
          *(_QWORD *)(v55 + 24) = v61;
        }
        if ( (unsigned __int64)(*(_QWORD *)(v59 + 16) - v61) <= 0x26 )
        {
          v59 = sub_16E7EE0(v59, "- Corruption of the global PassRegistry", 39);
          v64 = *(_BYTE **)(v59 + 24);
          if ( *(_BYTE **)(v59 + 16) != v64 )
            goto LABEL_45;
          goto LABEL_54;
        }
        v62 = _mm_load_si128((const __m128i *)&xmmword_3F553A0);
        *(_DWORD *)(v61 + 32) = 1936287589;
        *(_WORD *)(v61 + 36) = 29300;
        *(__m128i *)v61 = v62;
        v63 = _mm_load_si128((const __m128i *)&xmmword_3F553B0);
        *(_BYTE *)(v61 + 38) = 121;
        *(__m128i *)(v61 + 16) = v63;
        v64 = (_BYTE *)(*(_QWORD *)(v59 + 24) + 39LL);
        *(_QWORD *)(v59 + 24) = v64;
        if ( *(_BYTE **)(v59 + 16) == v64 )
        {
LABEL_54:
          sub_16E7EE0(v59, "\n", 1, v60, v56, v57, v71, v72, v73);
LABEL_28:
          if ( (char **)v77 == ++v28 )
            break;
        }
        else
        {
LABEL_45:
          *v64 = 10;
          ++v28;
          ++*(_QWORD *)(v59 + 24);
          if ( (char **)v77 == v28 )
            break;
        }
      }
      v6 = v73;
      v4 = v72;
      v3 = v71;
LABEL_7:
      v10 = (__int64 *)(*(__int64 (**)(void))(v81 + 72))();
      v82 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v3 + 80LL))(v3);
      if ( v82 == (*(unsigned int (__fastcall **)(__int64 *))(*v10 + 80))(v10) )
      {
        result = (__int64)sub_16185C0(*a1, v10);
        goto LABEL_4;
      }
      v83 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v3 + 80LL))(v3);
      if ( v83 > (*(int (__fastcall **)(__int64 *))(*v10 + 80))(v10) )
      {
        result = (__int64)sub_16185C0(*a1, v10);
        v80 = 1;
        goto LABEL_4;
      }
      ++v4;
      result = (*(__int64 (__fastcall **)(__int64 *))(*v10 + 8))(v10);
    }
    while ( (__int64 *)v6 != v4 );
LABEL_10:
    ;
  }
  while ( v80 );
  return result;
}
