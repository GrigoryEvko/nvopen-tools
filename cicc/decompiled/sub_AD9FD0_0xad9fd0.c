// Function: sub_AD9FD0
// Address: 0xad9fd0
//
__int64 __fastcall sub_AD9FD0(
        __int64 a1,
        unsigned __int8 *a2,
        __int64 *a3,
        __int64 a4,
        unsigned __int8 a5,
        __int64 a6,
        __int64 a7)
{
  bool v11; // zf
  __int64 v12; // rax
  __int64 v13; // r8
  int v15; // eax
  char v16; // dl
  int v17; // eax
  __int64 v18; // rax
  char *v19; // rax
  __int64 v20; // r8
  char *v21; // rcx
  size_t v22; // rdx
  __int64 *v23; // r13
  __int64 v24; // rcx
  __int64 v25; // rax
  __int64 v26; // r8
  char *v27; // rsi
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // rax
  __int64 v30; // rdi
  _BYTE *v31; // rax
  unsigned __int64 v32; // rax
  char v33; // bl
  __int64 *v34; // rax
  __m128i v35; // xmm0
  __m128i v36; // xmm1
  __int64 v37; // r12
  __int64 v38; // r12
  __int64 v39; // rax
  __int64 *v40; // rsi
  __int64 *v41; // rax
  __int64 v42; // rcx
  int v43; // edx
  char v44; // dl
  int v45; // eax
  __int64 v46; // rax
  __int64 v47; // [rsp+10h] [rbp-150h]
  int v48; // [rsp+1Ch] [rbp-144h]
  __int64 v49; // [rsp+20h] [rbp-140h]
  __int64 v50; // [rsp+20h] [rbp-140h]
  __int64 v51; // [rsp+20h] [rbp-140h]
  char *v52; // [rsp+20h] [rbp-140h]
  __int64 v53; // [rsp+28h] [rbp-138h]
  __int64 v54; // [rsp+28h] [rbp-138h]
  __int64 v55; // [rsp+28h] [rbp-138h]
  __int64 v56; // [rsp+28h] [rbp-138h]
  __int64 v57; // [rsp+28h] [rbp-138h]
  __int64 v58; // [rsp+28h] [rbp-138h]
  __int64 v60; // [rsp+30h] [rbp-130h]
  __int64 v61; // [rsp+30h] [rbp-130h]
  __int64 v62; // [rsp+30h] [rbp-130h]
  __int64 v63; // [rsp+30h] [rbp-130h]
  __int64 v64; // [rsp+30h] [rbp-130h]
  unsigned __int8 *v65; // [rsp+38h] [rbp-128h] BYREF
  __int64 v66; // [rsp+48h] [rbp-118h]
  void *src; // [rsp+50h] [rbp-110h] BYREF
  char *v68; // [rsp+58h] [rbp-108h]
  char *v69; // [rsp+60h] [rbp-100h]
  __int16 v70; // [rsp+70h] [rbp-F0h]
  __m128i v71; // [rsp+78h] [rbp-E8h] BYREF
  __m128i v72; // [rsp+88h] [rbp-D8h] BYREF
  __int64 v73; // [rsp+98h] [rbp-C8h]
  _BYTE *v74; // [rsp+A0h] [rbp-C0h] BYREF
  unsigned int v75; // [rsp+A8h] [rbp-B8h]
  __int64 v76; // [rsp+B0h] [rbp-B0h] BYREF
  unsigned int v77; // [rsp+B8h] [rbp-A8h]
  char v78; // [rsp+C0h] [rbp-A0h]
  _BYTE *v79; // [rsp+D0h] [rbp-90h] BYREF
  __m128i v80; // [rsp+D8h] [rbp-88h] BYREF
  __m128i v81; // [rsp+E8h] [rbp-78h]
  __int64 v82; // [rsp+F8h] [rbp-68h]
  _BYTE *v83; // [rsp+100h] [rbp-60h] BYREF
  unsigned int v84; // [rsp+108h] [rbp-58h]
  __int64 v85; // [rsp+110h] [rbp-50h] BYREF
  unsigned int v86; // [rsp+118h] [rbp-48h]
  char v87; // [rsp+120h] [rbp-40h]

  v11 = *(_BYTE *)(a6 + 32) == 0;
  v81.m128i_i8[8] = 0;
  v65 = a2;
  if ( !v11 )
  {
    v80.m128i_i32[0] = *(_DWORD *)(a6 + 8);
    if ( v80.m128i_i32[0] > 0x40u )
      sub_C43780(&v79, a6);
    else
      v79 = *(_BYTE **)a6;
    v81.m128i_i32[0] = *(_DWORD *)(a6 + 24);
    if ( v81.m128i_i32[0] > 0x40u )
      sub_C43780(&v80.m128i_u64[1], a6 + 16);
    else
      v80.m128i_i64[1] = *(_QWORD *)(a6 + 16);
    v81.m128i_i8[8] = 1;
  }
  v12 = sub_AAB960(a1, v65, (__int64)&v79, a3, a4);
  v13 = v12;
  if ( v81.m128i_i8[8] )
  {
    v81.m128i_i8[8] = 0;
    if ( v81.m128i_i32[0] > 0x40u && v80.m128i_i64[1] )
    {
      v53 = v12;
      j_j___libc_free_0_0(v80.m128i_i64[1]);
      v13 = v53;
    }
    if ( v80.m128i_i32[0] > 0x40u && v79 )
    {
      v54 = v13;
      j_j___libc_free_0_0(v79);
      v13 = v54;
    }
  }
  if ( !v13 )
  {
    v47 = *((_QWORD *)v65 + 1);
    if ( (unsigned int)*(unsigned __int8 *)(v47 + 8) - 17 > 1 )
    {
      v40 = &a3[a4];
      if ( v40 != a3 )
      {
        v41 = a3;
        while ( 1 )
        {
          v42 = *(_QWORD *)(*v41 + 8);
          v43 = *(unsigned __int8 *)(v42 + 8);
          if ( v43 == 17 )
          {
            v44 = 0;
            goto LABEL_93;
          }
          if ( v43 == 18 )
            break;
          if ( v40 == ++v41 )
            goto LABEL_17;
        }
        v44 = 1;
LABEL_93:
        v45 = *(_DWORD *)(v42 + 32);
        BYTE4(v79) = v44;
        LODWORD(v79) = v45;
        v46 = sub_BCE1B0(v47, v79);
        v13 = 0;
        v47 = v46;
      }
    }
LABEL_17:
    if ( a7 != v47 )
    {
      v15 = *(unsigned __int8 *)(v47 + 8);
      BYTE4(v66) = 0;
      v48 = 0;
      v16 = v15;
      if ( (unsigned int)(v15 - 17) <= 1 )
      {
        v17 = *(_DWORD *)(v47 + 32);
        BYTE4(v66) = v16 == 18;
        v48 = v17;
        LODWORD(v66) = v17;
      }
      v18 = a4 + 1;
      src = 0;
      v68 = 0;
      v69 = 0;
      if ( (unsigned __int64)(a4 + 1) > 0xFFFFFFFFFFFFFFFLL )
        sub_4262D8((__int64)"vector::reserve");
      if ( a4 == -1 )
      {
        sub_91DE00((__int64)&src, 0, &v65);
      }
      else
      {
        v55 = 8 * v18;
        v19 = (char *)sub_22077B0(8 * v18);
        v20 = v55;
        v21 = v19;
        v22 = v68 - (_BYTE *)src;
        if ( v68 - (_BYTE *)src > 0 )
        {
          v52 = (char *)memmove(v19, src, v22);
          j_j___libc_free_0(src, v69 - (_BYTE *)src);
          v21 = v52;
          v20 = v55;
        }
        src = v21;
        v69 = &v21[v20];
        if ( v21 )
          *(_QWORD *)v21 = v65;
        v68 = v21 + 8;
      }
      v23 = &a3[a4];
      v24 = a1 & 0xFFFFFFFFFFFFFFF9LL | 4;
      if ( v23 == a3 )
      {
LABEL_48:
        v33 = *(_BYTE *)(a6 + 32);
        v81.m128i_i8[8] = 0;
        if ( v33 )
        {
          v80.m128i_i32[0] = *(_DWORD *)(a6 + 8);
          if ( v80.m128i_i32[0] > 0x40u )
            sub_C43780(&v79, a6);
          else
            v79 = *(_BYTE **)a6;
          v81.m128i_i32[0] = *(_DWORD *)(a6 + 24);
          if ( v81.m128i_i32[0] > 0x40u )
            sub_C43780(&v80.m128i_u64[1], a6 + 16);
          else
            v80.m128i_i64[1] = *(_QWORD *)(a6 + 16);
          v81.m128i_i8[8] = 1;
        }
        LOBYTE(v70) = 34;
        v73 = a1;
        v71.m128i_i64[0] = (__int64)src;
        HIBYTE(v70) = a5;
        v71.m128i_i64[1] = (v68 - (_BYTE *)src) >> 3;
        v72 = 0u;
        v78 = 0;
        if ( v33 )
        {
          v78 = 1;
          v75 = v80.m128i_i32[0];
          v74 = v79;
          v77 = v81.m128i_i32[0];
          v76 = v80.m128i_i64[1];
        }
        v34 = (__int64 *)sub_BD5C60(v65, a5, src);
        v35 = _mm_loadu_si128(&v71);
        v36 = _mm_loadu_si128(&v72);
        v37 = *v34;
        v87 = 0;
        v80 = v35;
        LOWORD(v79) = v70;
        v38 = v37 + 2120;
        v81 = v36;
        v82 = v73;
        if ( v78 )
        {
          v84 = v75;
          if ( v75 > 0x40 )
            sub_C43780(&v83, &v74);
          else
            v83 = v74;
          v86 = v77;
          if ( v77 > 0x40 )
            sub_C43780(&v85, &v76);
          else
            v85 = v76;
          v87 = 1;
        }
        v39 = sub_AD4210(v38, v47, (__int16 *)&v79);
        v13 = v39;
        if ( v87 )
        {
          v87 = 0;
          if ( v86 > 0x40 && v85 )
          {
            v63 = v39;
            j_j___libc_free_0_0(v85);
            v13 = v63;
          }
          if ( v84 > 0x40 && v83 )
          {
            v64 = v13;
            j_j___libc_free_0_0(v83);
            v13 = v64;
          }
        }
        if ( v78 )
        {
          v78 = 0;
          if ( v77 > 0x40 && v76 )
          {
            v61 = v13;
            j_j___libc_free_0_0(v76);
            v13 = v61;
          }
          if ( v75 > 0x40 && v74 )
          {
            v62 = v13;
            j_j___libc_free_0_0(v74);
            v13 = v62;
          }
        }
        if ( src )
        {
          v60 = v13;
          j_j___libc_free_0(src, v69 - (_BYTE *)src);
          return v60;
        }
        return v13;
      }
      while ( 1 )
      {
        v30 = *a3;
        v79 = (_BYTE *)*a3;
        v26 = (v24 >> 1) & 3;
        if ( ((v24 >> 1) & 3) != 0 )
          break;
        if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v30 + 8) + 8LL) - 17 > 1 )
          goto LABEL_32;
        v50 = (v24 >> 1) & 3;
        v57 = v24;
        v31 = sub_AD7630(v30, 0, v22);
        v24 = v57;
        v26 = v50;
        v79 = v31;
        v27 = v68;
        if ( v68 != v69 )
        {
LABEL_33:
          if ( v27 )
          {
            *(_QWORD *)v27 = v79;
            v27 = v68;
          }
          v28 = v24 & 0xFFFFFFFFFFFFFFF8LL;
          v68 = v27 + 8;
          v29 = v24 & 0xFFFFFFFFFFFFFFF8LL;
          if ( !v24 )
            goto LABEL_45;
          goto LABEL_36;
        }
LABEL_44:
        v51 = v26;
        v58 = v24;
        sub_91DE00((__int64)&src, v27, &v79);
        v26 = v51;
        v28 = v58 & 0xFFFFFFFFFFFFFFF8LL;
        v29 = v58 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v58 )
          goto LABEL_45;
LABEL_36:
        if ( v26 == 2 )
        {
          if ( !v28 )
            goto LABEL_45;
LABEL_38:
          v22 = *(unsigned __int8 *)(v29 + 8);
          if ( (_BYTE)v22 != 16 )
            goto LABEL_46;
LABEL_39:
          v24 = *(_QWORD *)(v29 + 24) & 0xFFFFFFFFFFFFFFF9LL | 4;
LABEL_40:
          if ( v23 == ++a3 )
            goto LABEL_48;
        }
        else
        {
          if ( v26 == 1 && v28 )
          {
            v29 = *(_QWORD *)(v28 + 24);
            goto LABEL_38;
          }
LABEL_45:
          v29 = sub_BCBAE0(v28, *a3);
          v22 = *(unsigned __int8 *)(v29 + 8);
          if ( (_BYTE)v22 == 16 )
            goto LABEL_39;
LABEL_46:
          v32 = v29 & 0xFFFFFFFFFFFFFFF9LL;
          if ( (unsigned int)(unsigned __int8)v22 - 17 > 1 )
          {
            v24 = 0;
            if ( (_BYTE)v22 == 15 )
              v24 = v32;
            goto LABEL_40;
          }
          ++a3;
          v24 = v32 | 2;
          if ( v23 == a3 )
            goto LABEL_48;
        }
      }
      if ( v48 && (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v30 + 8) + 8LL) - 17 > 1 )
      {
        LODWORD(v66) = v48;
        v49 = (v24 >> 1) & 3;
        v56 = v24;
        v25 = sub_AD5E10(v66, (unsigned __int8 *)v30);
        v26 = v49;
        v24 = v56;
        v79 = (_BYTE *)v25;
      }
LABEL_32:
      v27 = v68;
      if ( v68 != v69 )
        goto LABEL_33;
      goto LABEL_44;
    }
  }
  return v13;
}
