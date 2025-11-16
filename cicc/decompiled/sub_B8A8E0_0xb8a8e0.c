// Function: sub_B8A8E0
// Address: 0xb8a8e0
//
__int64 __fastcall sub_B8A8E0(__int64 *a1, __int64 *a2)
{
  __int64 *v3; // r14
  __int64 *v4; // rbx
  __int64 result; // rax
  __int64 v6; // r13
  __int64 v7; // r12
  __int64 v8; // rdi
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r8
  __int64 v13; // rax
  size_t v14; // rdx
  __int64 v15; // r8
  const void *v16; // rsi
  _OWORD *v17; // rdi
  unsigned __int64 v18; // rax
  __m128i si128; // xmm0
  _BYTE *v20; // rax
  __int64 v21; // rax
  __m128i *v22; // rdx
  __int64 v23; // rdi
  __m128i v24; // xmm0
  _BYTE *v25; // rax
  __int64 v26; // rax
  __m128i *v27; // rdx
  __int64 v28; // rdi
  _BYTE *v29; // rax
  __int64 *v30; // rbx
  __int64 v31; // r14
  _BYTE *v32; // rax
  __int64 v33; // rax
  size_t v34; // rdx
  _BYTE *v35; // rdi
  const void *v36; // rsi
  _BYTE *v37; // rax
  __int64 v38; // rdi
  __int64 v39; // r13
  __int64 v40; // rdi
  _BYTE *v41; // rax
  __m128i *v42; // rdx
  __int64 v43; // rax
  _BYTE *v44; // rax
  __int64 v45; // rax
  _WORD *v46; // rdx
  __int64 v47; // rdi
  __int64 v48; // rdx
  __m128i v49; // xmm0
  _BYTE *v50; // rax
  __int64 v51; // rax
  _WORD *v52; // rdx
  __int64 v53; // rdi
  __int64 v54; // rdx
  __m128i v55; // xmm0
  __m128i v56; // xmm0
  _BYTE *v57; // rax
  _BYTE *v58; // rdx
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // rax
  _OWORD *v63; // rdx
  __int64 *v64; // [rsp+0h] [rbp-70h]
  __int64 *v65; // [rsp+8h] [rbp-68h]
  __int64 v66; // [rsp+10h] [rbp-60h]
  size_t v67; // [rsp+18h] [rbp-58h]
  __int64 v68; // [rsp+18h] [rbp-58h]
  __int64 v69; // [rsp+20h] [rbp-50h]
  __int64 v70; // [rsp+20h] [rbp-50h]
  size_t v71; // [rsp+20h] [rbp-50h]
  __int64 v72; // [rsp+28h] [rbp-48h]
  char v73; // [rsp+37h] [rbp-39h]
  __int64 v74; // [rsp+38h] [rbp-38h]
  int v75; // [rsp+38h] [rbp-38h]
  int v76; // [rsp+38h] [rbp-38h]

  v3 = a2;
  v72 = sub_B873F0(*a1, a2);
  do
  {
    v4 = *(__int64 **)v72;
    result = *(_QWORD *)v72 + 8LL * *(unsigned int *)(v72 + 8);
    if ( result == *(_QWORD *)v72 )
      break;
    v73 = 0;
    v6 = *(_QWORD *)v72 + 8LL * *(unsigned int *)(v72 + 8);
    do
    {
      while ( 1 )
      {
        v7 = *v4;
        result = sub_B811E0(*a1, *v4);
        if ( !result )
          break;
LABEL_4:
        if ( (__int64 *)v6 == ++v4 )
          goto LABEL_10;
      }
      v8 = *a1;
      v74 = sub_B85AD0(*a1, v7);
      if ( v74 )
        goto LABEL_7;
      v10 = sub_C5F790(v8);
      v11 = *(_QWORD *)(v10 + 32);
      v12 = v10;
      if ( (unsigned __int64)(*(_QWORD *)(v10 + 24) - v11) <= 5 )
      {
        v12 = sub_CB6200(v10, "Pass '", 6);
      }
      else
      {
        *(_DWORD *)v11 = 1936941392;
        *(_WORD *)(v11 + 4) = 10016;
        *(_QWORD *)(v10 + 32) += 6LL;
      }
      v69 = v12;
      v13 = (*(__int64 (__fastcall **)(__int64 *))(*v3 + 16))(v3);
      v15 = v69;
      v16 = (const void *)v13;
      v17 = *(_OWORD **)(v69 + 32);
      v18 = *(_QWORD *)(v69 + 24) - (_QWORD)v17;
      if ( v14 > v18 )
      {
        v61 = sub_CB6200(v69, v16, v14);
        v17 = *(_OWORD **)(v61 + 32);
        v15 = v61;
        if ( *(_QWORD *)(v61 + 24) - (_QWORD)v17 > 0x14u )
          goto LABEL_20;
      }
      else
      {
        if ( v14 )
        {
          v68 = v69;
          v71 = v14;
          memcpy(v17, v16, v14);
          v15 = v68;
          v62 = *(_QWORD *)(v68 + 24);
          v63 = (_OWORD *)(*(_QWORD *)(v68 + 32) + v71);
          *(_QWORD *)(v68 + 32) = v63;
          v17 = v63;
          v18 = v62 - (_QWORD)v63;
        }
        if ( v18 > 0x14 )
        {
LABEL_20:
          si128 = _mm_load_si128((const __m128i *)&xmmword_3F55310);
          *((_DWORD *)v17 + 4) = 1684372073;
          *((_BYTE *)v17 + 20) = 46;
          *v17 = si128;
          v20 = (_BYTE *)(*(_QWORD *)(v15 + 32) + 21LL);
          *(_QWORD *)(v15 + 32) = v20;
          if ( *(_BYTE **)(v15 + 24) != v20 )
            goto LABEL_21;
          goto LABEL_72;
        }
      }
      v17 = (_OWORD *)v15;
      v15 = sub_CB6200(v15, "' is not initialized.", 21);
      v20 = *(_BYTE **)(v15 + 32);
      if ( *(_BYTE **)(v15 + 24) != v20 )
      {
LABEL_21:
        *v20 = 10;
        ++*(_QWORD *)(v15 + 32);
        goto LABEL_22;
      }
LABEL_72:
      v17 = (_OWORD *)v15;
      sub_CB6200(v15, "\n", 1);
LABEL_22:
      v21 = sub_C5F790(v17);
      v22 = *(__m128i **)(v21 + 32);
      v23 = v21;
      if ( *(_QWORD *)(v21 + 24) - (_QWORD)v22 <= 0x2Au )
      {
        v23 = sub_CB6200(v21, "Verify if there is a pass dependency cycle.", 43);
        v25 = *(_BYTE **)(v23 + 32);
      }
      else
      {
        v24 = _mm_load_si128((const __m128i *)&xmmword_3F55320);
        qmemcpy(&v22[2], "ency cycle.", 11);
        *v22 = v24;
        v22[1] = _mm_load_si128((const __m128i *)&xmmword_3F55330);
        v25 = (_BYTE *)(*(_QWORD *)(v21 + 32) + 43LL);
        *(_QWORD *)(v23 + 32) = v25;
      }
      if ( *(_BYTE **)(v23 + 24) == v25 )
      {
        sub_CB6200(v23, "\n", 1);
      }
      else
      {
        *v25 = 10;
        ++*(_QWORD *)(v23 + 32);
      }
      v26 = sub_C5F790(v23);
      v27 = *(__m128i **)(v26 + 32);
      v28 = v26;
      if ( *(_QWORD *)(v26 + 24) - (_QWORD)v27 <= 0xFu )
      {
        v28 = sub_CB6200(v26, "Required Passes:", 16);
        v29 = *(_BYTE **)(v28 + 32);
      }
      else
      {
        *v27 = _mm_load_si128((const __m128i *)&xmmword_3F55340);
        v29 = (_BYTE *)(*(_QWORD *)(v26 + 32) + 16LL);
        *(_QWORD *)(v28 + 32) = v29;
      }
      if ( *(_BYTE **)(v28 + 24) == v29 )
      {
        sub_CB6200(v28, "\n", 1);
      }
      else
      {
        *v29 = 10;
        ++*(_QWORD *)(v28 + 32);
      }
      v70 = *(_QWORD *)v72 + 8LL * *(unsigned int *)(v72 + 8);
      if ( v70 != *(_QWORD *)v72 )
      {
        v65 = v4;
        v30 = *(__int64 **)v72;
        v66 = v6;
        v64 = v3;
        while ( 1 )
        {
          if ( v7 == *v30 )
          {
LABEL_57:
            v6 = v66;
            v4 = v65;
            v3 = v64;
            break;
          }
          v38 = *a1;
          v39 = sub_B811E0(*a1, *v30);
          if ( v39 )
          {
            v31 = sub_C5F790(v38);
            v32 = *(_BYTE **)(v31 + 32);
            if ( *(_BYTE **)(v31 + 24) == v32 )
            {
              v31 = sub_CB6200(v31, "\t", 1);
            }
            else
            {
              *v32 = 9;
              ++*(_QWORD *)(v31 + 32);
            }
            v33 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v39 + 16LL))(v39);
            v35 = *(_BYTE **)(v31 + 32);
            v36 = (const void *)v33;
            v37 = *(_BYTE **)(v31 + 24);
            if ( v37 - v35 < v34 )
            {
              v31 = sub_CB6200(v31, v36, v34);
              v37 = *(_BYTE **)(v31 + 24);
              v35 = *(_BYTE **)(v31 + 32);
            }
            else if ( v34 )
            {
              v67 = v34;
              memcpy(v35, v36, v34);
              v58 = (_BYTE *)(*(_QWORD *)(v31 + 32) + v67);
              *(_QWORD *)(v31 + 32) = v58;
              v37 = *(_BYTE **)(v31 + 24);
              v35 = v58;
            }
            if ( v37 == v35 )
            {
              sub_CB6200(v31, "\n", 1);
            }
            else
            {
              *v35 = 10;
              ++*(_QWORD *)(v31 + 32);
            }
            goto LABEL_39;
          }
          v40 = sub_C5F790(v38);
          v41 = *(_BYTE **)(v40 + 32);
          if ( *(_BYTE **)(v40 + 24) == v41 )
          {
            v40 = sub_CB6200(v40, "\t", 1);
            v42 = *(__m128i **)(v40 + 32);
            if ( *(_QWORD *)(v40 + 24) - (_QWORD)v42 > 0x2Fu )
            {
LABEL_44:
              *v42 = _mm_load_si128((const __m128i *)&xmmword_3F55350);
              v42[1] = _mm_load_si128((const __m128i *)&xmmword_3F55360);
              v42[2] = _mm_load_si128((const __m128i *)&xmmword_3F55370);
              v44 = (_BYTE *)(*(_QWORD *)(v40 + 32) + 48LL);
              *(_QWORD *)(v40 + 32) = v44;
              if ( *(_BYTE **)(v40 + 24) != v44 )
                goto LABEL_45;
              goto LABEL_63;
            }
          }
          else
          {
            *v41 = 9;
            v42 = (__m128i *)(*(_QWORD *)(v40 + 32) + 1LL);
            v43 = *(_QWORD *)(v40 + 24);
            *(_QWORD *)(v40 + 32) = v42;
            if ( (unsigned __int64)(v43 - (_QWORD)v42) > 0x2F )
              goto LABEL_44;
          }
          v40 = sub_CB6200(v40, "Error: Required pass not found! Possible causes:", 48);
          v44 = *(_BYTE **)(v40 + 32);
          if ( *(_BYTE **)(v40 + 24) != v44 )
          {
LABEL_45:
            *v44 = 10;
            ++*(_QWORD *)(v40 + 32);
            goto LABEL_46;
          }
LABEL_63:
          sub_CB6200(v40, "\n", 1);
LABEL_46:
          v45 = sub_C5F790(v40);
          v46 = *(_WORD **)(v45 + 32);
          v47 = v45;
          if ( *(_QWORD *)(v45 + 24) - (_QWORD)v46 <= 1u )
          {
            v60 = sub_CB6200(v45, "\t\t", 2);
            v48 = *(_QWORD *)(v60 + 32);
            v47 = v60;
          }
          else
          {
            *v46 = 2313;
            v48 = *(_QWORD *)(v45 + 32) + 2LL;
            *(_QWORD *)(v45 + 32) = v48;
          }
          if ( (unsigned __int64)(*(_QWORD *)(v47 + 24) - v48) <= 0x2D )
          {
            v47 = sub_CB6200(v47, "- Pass misconfiguration (e.g.: missing macros)", 46);
            v50 = *(_BYTE **)(v47 + 32);
          }
          else
          {
            v49 = _mm_load_si128((const __m128i *)&xmmword_3F55380);
            qmemcpy((void *)(v48 + 32), "issing macros)", 14);
            *(__m128i *)v48 = v49;
            *(__m128i *)(v48 + 16) = _mm_load_si128((const __m128i *)&xmmword_3F55390);
            v50 = (_BYTE *)(*(_QWORD *)(v47 + 32) + 46LL);
            *(_QWORD *)(v47 + 32) = v50;
          }
          if ( *(_BYTE **)(v47 + 24) == v50 )
          {
            sub_CB6200(v47, "\n", 1);
          }
          else
          {
            *v50 = 10;
            ++*(_QWORD *)(v47 + 32);
          }
          v51 = sub_C5F790(v47);
          v52 = *(_WORD **)(v51 + 32);
          v53 = v51;
          if ( *(_QWORD *)(v51 + 24) - (_QWORD)v52 <= 1u )
          {
            v59 = sub_CB6200(v51, "\t\t", 2);
            v54 = *(_QWORD *)(v59 + 32);
            v53 = v59;
          }
          else
          {
            *v52 = 2313;
            v54 = *(_QWORD *)(v51 + 32) + 2LL;
            *(_QWORD *)(v51 + 32) = v54;
          }
          if ( (unsigned __int64)(*(_QWORD *)(v53 + 24) - v54) <= 0x26 )
          {
            v53 = sub_CB6200(v53, "- Corruption of the global PassRegistry", 39);
            v57 = *(_BYTE **)(v53 + 32);
            if ( *(_BYTE **)(v53 + 24) != v57 )
              goto LABEL_56;
            goto LABEL_65;
          }
          v55 = _mm_load_si128((const __m128i *)&xmmword_3F553A0);
          *(_DWORD *)(v54 + 32) = 1936287589;
          *(_WORD *)(v54 + 36) = 29300;
          *(__m128i *)v54 = v55;
          v56 = _mm_load_si128((const __m128i *)&xmmword_3F553B0);
          *(_BYTE *)(v54 + 38) = 121;
          *(__m128i *)(v54 + 16) = v56;
          v57 = (_BYTE *)(*(_QWORD *)(v53 + 32) + 39LL);
          *(_QWORD *)(v53 + 32) = v57;
          if ( *(_BYTE **)(v53 + 24) == v57 )
          {
LABEL_65:
            sub_CB6200(v53, "\n", 1);
LABEL_39:
            if ( (__int64 *)v70 == ++v30 )
              goto LABEL_57;
          }
          else
          {
LABEL_56:
            *v57 = 10;
            ++v30;
            ++*(_QWORD *)(v53 + 32);
            if ( (__int64 *)v70 == v30 )
              goto LABEL_57;
          }
        }
      }
LABEL_7:
      v9 = (*(__int64 (**)(void))(v74 + 48))();
      v75 = (*(__int64 (__fastcall **)(__int64 *))(*v3 + 80))(v3);
      if ( v75 == (*(unsigned int (__fastcall **)(__int64))(*(_QWORD *)v9 + 80LL))(v9) )
      {
        result = sub_B8B080(*a1, v9);
        goto LABEL_4;
      }
      v76 = (*(__int64 (__fastcall **)(__int64 *))(*v3 + 80))(v3);
      if ( v76 > (*(int (__fastcall **)(__int64))(*(_QWORD *)v9 + 80LL))(v9) )
      {
        result = sub_B8B080(*a1, v9);
        v73 = 1;
        goto LABEL_4;
      }
      ++v4;
      result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v9 + 8LL))(v9);
    }
    while ( (__int64 *)v6 != v4 );
LABEL_10:
    ;
  }
  while ( v73 );
  return result;
}
