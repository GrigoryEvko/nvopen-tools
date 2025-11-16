// Function: sub_255E680
// Address: 0x255e680
//
__int64 __fastcall sub_255E680(_QWORD *a1, __m128i *a2, __int64 a3, char a4)
{
  __int64 *v5; // r12
  __int64 v7; // rax
  unsigned int v8; // r14d
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rcx
  unsigned int v13; // r14d
  __int64 v15; // rbx
  __int64 v16; // rax
  __int64 v17; // r15
  unsigned __int64 v18; // rbx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  unsigned __int64 *v23; // rax
  __int64 v24; // rbx
  __int64 *v25; // r15
  __int64 v26; // rbx
  __int64 *v27; // rdx
  __int64 *v28; // rax
  __int64 v29; // rsi
  __int64 v30; // rbx
  __int64 *v31; // rax
  _QWORD *v32; // r8
  __int64 v33; // rdi
  __int64 v34; // rsi
  __int64 v35; // rax
  unsigned __int64 v36; // r8
  __int64 v37; // rdi
  __int64 v38; // rsi
  __int64 v39; // rax
  unsigned __int64 v40; // r8
  __int64 v41; // rax
  __int64 v42; // rsi
  __int64 v43; // rdi
  unsigned __int64 v44; // r8
  __int64 v45; // rdx
  __int64 v46; // rdi
  __int64 v47; // rax
  unsigned __int64 v48; // r8
  unsigned __int64 v49; // rdi
  __int64 *v50; // rax
  __int64 v51; // rax
  unsigned __int64 v52; // rdx
  __int64 v53; // rax
  __int64 v54; // rdi
  unsigned __int8 *v55; // rax
  char v56; // al
  __int64 *v57; // [rsp+8h] [rbp-148h]
  unsigned __int64 v58; // [rsp+20h] [rbp-130h]
  __int64 *v59; // [rsp+20h] [rbp-130h]
  char v60; // [rsp+3Fh] [rbp-111h] BYREF
  __int64 v61; // [rsp+40h] [rbp-110h] BYREF
  __int64 v62; // [rsp+48h] [rbp-108h] BYREF
  _DWORD *v63; // [rsp+50h] [rbp-100h] BYREF
  __int64 v64; // [rsp+58h] [rbp-F8h]
  _DWORD v65[4]; // [rsp+60h] [rbp-F0h] BYREF
  _QWORD *v66; // [rsp+70h] [rbp-E0h] BYREF
  __int64 *v67; // [rsp+78h] [rbp-D8h]
  __int64 *v68; // [rsp+80h] [rbp-D0h]
  __int64 *v69; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v70; // [rsp+98h] [rbp-B8h]
  _BYTE v71[48]; // [rsp+A0h] [rbp-B0h] BYREF
  __m128i v72; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v73; // [rsp+E0h] [rbp-70h]
  __int64 v74; // [rsp+E8h] [rbp-68h]
  __int64 v75; // [rsp+F0h] [rbp-60h]
  __int64 v76; // [rsp+F8h] [rbp-58h]
  __int64 v77; // [rsp+100h] [rbp-50h]
  __int64 v78; // [rsp+108h] [rbp-48h]
  __int16 v79; // [rsp+110h] [rbp-40h]

  v5 = (__int64 *)a2;
  v63 = v65;
  v64 = 0x200000001LL;
  v65[0] = 43;
  v7 = sub_250D180(a2->m128i_i64, (__int64)a2);
  if ( (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17 <= 1 )
    v7 = **(_QWORD **)(v7 + 16);
  v8 = *(_DWORD *)(v7 + 8);
  v9 = sub_25096F0(a2);
  if ( sub_B2F070(v9, v8 >> 8) )
  {
    v12 = (unsigned int)v64;
  }
  else
  {
    v51 = (unsigned int)v64;
    v52 = (unsigned int)v64 + 1LL;
    if ( v52 > HIDWORD(v64) )
    {
      sub_C8D5F0((__int64)&v63, v65, v52, 4u, v10, v11);
      v51 = (unsigned int)v64;
    }
    v63[v51] = 90;
    v12 = (unsigned int)(v64 + 1);
    LODWORD(v64) = v64 + 1;
  }
  v13 = sub_2516400((__int64)a1, a2, (__int64)v63, v12, a4, 43);
  if ( !(_BYTE)v13 )
  {
    v15 = a1[26];
    v61 = 0;
    v62 = 0;
    v16 = sub_25096F0(a2);
    v17 = v16;
    if ( v16 && !sub_B2FC80(v16) )
    {
      v53 = sub_2554D30(*(_QWORD *)(v15 + 240), v17, 0);
      v54 = *(_QWORD *)(v15 + 240);
      a2 = (__m128i *)v17;
      v61 = v53;
      v62 = sub_255E580(v54, v17, 0);
    }
    v69 = (__int64 *)v71;
    v70 = 0x300000000LL;
    if ( (unsigned __int8)sub_2509800(v5) == 2 )
    {
      v60 = 0;
      LODWORD(v66) = 1;
      v55 = sub_250CBE0(v5, (__int64)a2);
      v72.m128i_i64[0] = (__int64)&v69;
      v56 = sub_2526260(
              (__int64)a1,
              (__int64 (__fastcall *)(__int64, unsigned __int64, __int64))sub_253B0B0,
              (__int64)&v72,
              (unsigned __int64)v55,
              0,
              &v60,
              (int *)&v66,
              1,
              0,
              1u);
      v24 = (unsigned int)v70;
      if ( !v56 )
      {
        v13 = 0;
        goto LABEL_28;
      }
    }
    else
    {
      v58 = sub_250D070(v5);
      v18 = sub_2509740(v5);
      v21 = (unsigned int)v70;
      v22 = (unsigned int)v70 + 1LL;
      if ( v22 > HIDWORD(v70) )
      {
        sub_C8D5F0((__int64)&v69, v71, v22, 0x10u, v19, v20);
        v21 = (unsigned int)v70;
      }
      v23 = (unsigned __int64 *)&v69[2 * v21];
      v23[1] = v18;
      *v23 = v58;
      v24 = (unsigned int)(v70 + 1);
      LODWORD(v70) = v70 + 1;
    }
    v25 = v69;
    v26 = 16 * v24;
    v27 = &v61;
    v66 = a1;
    v67 = &v61;
    v28 = &v69[(unsigned __int64)v26 / 8];
    v29 = v26 >> 4;
    v30 = v26 >> 6;
    v57 = v28;
    v31 = &v62;
    v68 = &v62;
    if ( v30 )
    {
      v32 = a1;
      v59 = &v69[8 * v30];
      while ( 1 )
      {
        v45 = *v27;
        v46 = *v25;
        v47 = *v31;
        v48 = *(_QWORD *)(v32[26] + 104LL);
        v76 = v25[1];
        v74 = v45;
        v79 = 257;
        v72 = (__m128i)v48;
        v73 = 0;
        v75 = v47;
        v77 = 0;
        v78 = 0;
        if ( !(unsigned __int8)sub_9B6260(v46, &v72, 0) )
          goto LABEL_24;
        v33 = v25[2];
        v34 = *v67;
        v35 = v25[3];
        v36 = *(_QWORD *)(v66[26] + 104LL);
        v75 = *v68;
        v74 = v34;
        v72 = (__m128i)v36;
        v73 = 0;
        v76 = v35;
        v77 = 0;
        v78 = 0;
        v79 = 257;
        if ( !(unsigned __int8)sub_9B6260(v33, &v72, 0) )
        {
          v25 += 2;
          goto LABEL_24;
        }
        v37 = v25[4];
        v38 = *v67;
        v39 = v25[5];
        v40 = *(_QWORD *)(v66[26] + 104LL);
        v75 = *v68;
        v74 = v38;
        v79 = 257;
        v72 = (__m128i)v40;
        v73 = 0;
        v76 = v39;
        v77 = 0;
        v78 = 0;
        if ( !(unsigned __int8)sub_9B6260(v37, &v72, 0) )
        {
          v25 += 4;
          goto LABEL_24;
        }
        v41 = v25[7];
        v42 = *v67;
        v43 = v25[6];
        v44 = *(_QWORD *)(v66[26] + 104LL);
        v75 = *v68;
        v74 = v42;
        v76 = v41;
        v72 = (__m128i)v44;
        v73 = 0;
        v77 = 0;
        v78 = 0;
        v79 = 257;
        if ( !(unsigned __int8)sub_9B6260(v43, &v72, 0) )
        {
          v25 += 6;
          goto LABEL_24;
        }
        v25 += 8;
        if ( v59 == v25 )
        {
          v29 = ((char *)v57 - (char *)v25) >> 4;
          break;
        }
        v31 = v68;
        v27 = v67;
        v32 = v66;
      }
    }
    if ( v29 != 2 )
    {
      if ( v29 != 3 )
      {
        if ( v29 != 1 )
          goto LABEL_25;
LABEL_39:
        if ( !(unsigned __int8)sub_2535750(&v66, *v25, v25[1]) )
          goto LABEL_25;
LABEL_24:
        if ( v57 == v25 )
        {
LABEL_25:
          v49 = *v5 & 0xFFFFFFFFFFFFFFFCLL;
          if ( (*v5 & 3) == 3 )
            v49 = *(_QWORD *)(v49 + 24);
          v50 = (__int64 *)sub_BD5C60(v49);
          v13 = 1;
          v72.m128i_i64[0] = sub_A778C0(v50, 43, 0);
          sub_2516380((__int64)a1, v5, (__int64)&v72, 1, 0);
        }
LABEL_28:
        if ( v69 != (__int64 *)v71 )
          _libc_free((unsigned __int64)v69);
        goto LABEL_6;
      }
      if ( (unsigned __int8)sub_2535750(&v66, *v25, v25[1]) )
        goto LABEL_24;
      v25 += 2;
    }
    if ( (unsigned __int8)sub_2535750(&v66, *v25, v25[1]) )
      goto LABEL_24;
    v25 += 2;
    goto LABEL_39;
  }
LABEL_6:
  if ( v63 != v65 )
    _libc_free((unsigned __int64)v63);
  return v13;
}
