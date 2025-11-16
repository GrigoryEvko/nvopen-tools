// Function: sub_329ECD0
// Address: 0x329ecd0
//
__int64 __fastcall sub_329ECD0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        char a6,
        __int128 a7,
        __int128 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11,
        __int64 a12,
        int a13,
        int a14,
        __int64 a15)
{
  int v15; // r13d
  __int64 v16; // rbx
  unsigned __int16 *v18; // rdx
  __int64 v19; // r15
  int v20; // eax
  __int64 v21; // r12
  __int64 v22; // r15
  __int64 v23; // rdx
  __int64 v24; // rdx
  int v25; // r9d
  char v26; // r13
  int v27; // ebx
  __int64 v28; // rdi
  __int128 *v29; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  int v34; // r9d
  int v35; // r9d
  __int64 v36; // rax
  __int64 v37; // rdx
  __m128i v38; // xmm1
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rdx
  int v42; // ecx
  bool v43; // zf
  char v44; // r12
  char v45; // r12
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rdx
  int v49; // esi
  __int64 v50; // rcx
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // rdx
  int v54; // esi
  __int64 v55; // rcx
  __int128 v56; // [rsp-30h] [rbp-170h]
  __int64 v57; // [rsp+0h] [rbp-140h]
  int v59; // [rsp+18h] [rbp-128h]
  unsigned int v61; // [rsp+20h] [rbp-120h]
  __int64 v62; // [rsp+28h] [rbp-118h]
  __int64 v63; // [rsp+30h] [rbp-110h]
  __int128 v64; // [rsp+40h] [rbp-100h]
  __int128 v65; // [rsp+50h] [rbp-F0h]
  __m128i v66; // [rsp+70h] [rbp-D0h]
  int v67; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v68; // [rsp+88h] [rbp-B8h]
  __int128 v69; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v70; // [rsp+A0h] [rbp-A0h]
  __int64 v71; // [rsp+A8h] [rbp-98h]
  int v72; // [rsp+B0h] [rbp-90h] BYREF
  __int128 *v73; // [rsp+B8h] [rbp-88h]
  unsigned __int64 v74; // [rsp+C0h] [rbp-80h]
  unsigned int v75; // [rsp+C8h] [rbp-78h]
  char v76; // [rsp+D4h] [rbp-6Ch]
  int v77; // [rsp+E0h] [rbp-60h] BYREF
  __int64 v78; // [rsp+E8h] [rbp-58h]
  int v79; // [rsp+F0h] [rbp-50h]
  unsigned __int64 v80; // [rsp+F8h] [rbp-48h] BYREF
  unsigned int v81; // [rsp+100h] [rbp-40h]
  int v82; // [rsp+108h] [rbp-38h]
  char v83; // [rsp+10Ch] [rbp-34h]

  *((_QWORD *)&v64 + 1) = a3;
  *(_QWORD *)&v64 = a2;
  v63 = a9;
  *((_QWORD *)&v65 + 1) = a5;
  v15 = a3;
  v62 = a12;
  *(_QWORD *)&v65 = a4;
  v16 = a11;
  v59 = a5;
  v18 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3);
  v19 = *((_QWORD *)v18 + 1);
  v20 = *v18;
  v21 = a10;
  v57 = a11;
  LOWORD(v67) = v20;
  v68 = v19;
  if ( (_WORD)v20 )
  {
    if ( (unsigned __int16)(v20 - 17) > 0xD3u )
    {
      LOWORD(v77) = v20;
      v78 = v19;
      goto LABEL_4;
    }
    LOWORD(v20) = word_4456580[v20 - 1];
    v23 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v67) )
    {
      v78 = v19;
      LOWORD(v77) = 0;
      goto LABEL_9;
    }
    LOWORD(v20) = sub_3009970((__int64)&v67, a2, v31, v32, v33);
  }
  LOWORD(v77) = v20;
  v78 = v23;
  if ( !(_WORD)v20 )
  {
LABEL_9:
    v70 = sub_3007260((__int64)&v77);
    LODWORD(v22) = v70;
    v71 = v24;
    goto LABEL_10;
  }
LABEL_4:
  if ( (_WORD)v20 == 1 || (unsigned __int16)(v20 - 504) <= 7u )
    BUG();
  v22 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v20 - 16];
LABEL_10:
  v26 = sub_3277840(v63, v21, v16, v62, v22, *a1, (v59 == v15) & (unsigned __int8)(a4 == a2));
  if ( !v26 )
  {
    if ( !(_DWORD)v22 )
      return 0;
    if ( a13 != 195 )
      return 0;
    v61 = v22 - 1;
    if ( ((unsigned int)v22 & ((_DWORD)v22 - 1)) != 0 )
      return 0;
    v76 = 0;
    *(_QWORD *)&v69 = 0;
    DWORD2(v69) = 0;
    v72 = 192;
    v73 = &v69;
    v75 = 64;
    v74 = 1;
    if ( !sub_329EC40(v65, *((__int64 *)&v65 + 1), 0, (__int64)&v72) )
    {
      if ( v75 > 0x40 && v74 )
        j_j___libc_free_0_0(v74);
      goto LABEL_27;
    }
    v43 = *(_DWORD *)(v16 + 24) == 188;
    v79 = v21;
    v77 = 188;
    v78 = v63;
    v81 = 64;
    v80 = v61;
    v83 = 0;
    if ( !v43 )
    {
      if ( v75 <= 0x40 )
        goto LABEL_27;
      v44 = 0;
      goto LABEL_49;
    }
    v46 = *(_QWORD *)(v16 + 40);
    v47 = *(_QWORD *)v46;
    if ( v63 )
    {
      if ( *(_DWORD *)(v46 + 8) != (_DWORD)v21 || v47 != v63 )
      {
        v48 = *(_QWORD *)(v46 + 40);
        v49 = *(_DWORD *)(v46 + 48);
        v50 = v63;
LABEL_64:
        if ( v48 != v50 || v79 != v49 )
          goto LABEL_65;
LABEL_85:
        if ( !(unsigned __int8)sub_32657E0((__int64)&v80, *(_QWORD *)v46) )
          goto LABEL_65;
        goto LABEL_86;
      }
    }
    else if ( !v47 )
    {
      v48 = *(_QWORD *)(v46 + 40);
      goto LABEL_84;
    }
    if ( (unsigned __int8)sub_32657E0((__int64)&v80, *(_QWORD *)(v46 + 40)) )
    {
LABEL_86:
      if ( !v83 || v82 == (v82 & *(_DWORD *)(v16 + 28)) )
      {
        v44 = sub_328A020(a1[1], 0xC3u, v67, v68, 0);
        goto LABEL_66;
      }
LABEL_65:
      v44 = 0;
LABEL_66:
      if ( v81 > 0x40 && v80 )
        j_j___libc_free_0_0(v80);
      if ( v75 <= 0x40 )
      {
LABEL_51:
        if ( v44 )
          return sub_340F900(*a1, 195, a15, v67, v68, v34, v64, v69, a7);
LABEL_27:
        v76 = 0;
        v72 = 190;
        v75 = 64;
        v73 = &v69;
        v74 = 1;
        if ( !sub_329EC40(v64, *((__int64 *)&v64 + 1), 0, (__int64)&v72) )
        {
          if ( v75 > 0x40 && v74 )
            j_j___libc_free_0_0(v74);
          goto LABEL_31;
        }
        v77 = 188;
        v78 = v16;
        v79 = v62;
        v81 = 64;
        v80 = v61;
        v83 = 0;
        if ( *(_DWORD *)(v63 + 24) != 188 )
        {
          if ( v75 <= 0x40 )
            goto LABEL_31;
          v45 = 0;
          goto LABEL_56;
        }
        v51 = *(_QWORD *)(v63 + 40);
        v52 = *(_QWORD *)v51;
        if ( v16 )
        {
          if ( *(_DWORD *)(v51 + 8) != (_DWORD)v62 || v52 != v16 )
          {
            v53 = *(_QWORD *)(v51 + 40);
            v54 = *(_DWORD *)(v51 + 48);
            v55 = v16;
LABEL_75:
            if ( v55 != v53 || v54 != v79 )
              goto LABEL_76;
LABEL_99:
            if ( !(unsigned __int8)sub_32657E0((__int64)&v80, *(_QWORD *)v51) )
              goto LABEL_76;
            goto LABEL_100;
          }
        }
        else if ( !v52 )
        {
          v53 = *(_QWORD *)(v51 + 40);
          goto LABEL_98;
        }
        if ( (unsigned __int8)sub_32657E0((__int64)&v80, *(_QWORD *)(v51 + 40)) )
        {
LABEL_100:
          if ( !v83 || v82 == (v82 & *(_DWORD *)(v63 + 28)) )
          {
            v45 = sub_328A020(a1[1], 0xC4u, v67, v68, 0);
            goto LABEL_77;
          }
LABEL_76:
          v45 = 0;
LABEL_77:
          if ( v81 > 0x40 && v80 )
            j_j___libc_free_0_0(v80);
          if ( v75 <= 0x40 )
          {
LABEL_58:
            if ( v45 )
              return sub_340F900(*a1, 196, a15, v67, v68, v35, v69, v65, a8);
LABEL_31:
            if ( *(_DWORD *)(a2 + 24) != 56 )
              return 0;
            v36 = *(_QWORD *)(a2 + 40);
            v66 = _mm_loadu_si128((const __m128i *)v36);
            *(_QWORD *)&v69 = *(_QWORD *)v36;
            DWORD2(v69) = v66.m128i_i32[2];
            v37 = *(_QWORD *)(v36 + 40);
            if ( v37 != (_QWORD)v69 || *(_DWORD *)(v36 + 48) != DWORD2(v69) )
            {
              v38 = _mm_loadu_si128((const __m128i *)(v36 + 40));
              *(_QWORD *)&v69 = *(_QWORD *)(v36 + 40);
              DWORD2(v69) = v38.m128i_i32[2];
              if ( v37 != *(_QWORD *)v36 || *(_DWORD *)(v36 + 8) != v38.m128i_i32[2] )
                return 0;
            }
            v77 = 188;
            v78 = v16;
            v79 = v62;
            v81 = 64;
            v80 = v61;
            v83 = 0;
            if ( *(_DWORD *)(v63 + 24) != 188 )
              return 0;
            v39 = *(_QWORD *)(v63 + 40);
            v40 = *(_QWORD *)v39;
            if ( v16 )
            {
              if ( *(_DWORD *)(v39 + 8) != (_DWORD)v62 || v40 != v16 )
              {
                v41 = *(_QWORD *)(v39 + 40);
                v42 = *(_DWORD *)(v39 + 48);
LABEL_40:
                if ( v41 != v57 || v79 != v42 )
                  goto LABEL_41;
                goto LABEL_111;
              }
            }
            else if ( !v40 )
            {
              v41 = *(_QWORD *)(v39 + 40);
              goto LABEL_110;
            }
            if ( (unsigned __int8)sub_32657E0((__int64)&v80, *(_QWORD *)(v39 + 40)) )
            {
LABEL_112:
              if ( !v83 || v82 == (v82 & *(_DWORD *)(v63 + 28)) )
              {
                v26 = sub_328A020(a1[1], 0xC4u, v67, v68, 0);
                if ( v81 <= 0x40 )
                  goto LABEL_44;
                goto LABEL_42;
              }
LABEL_41:
              if ( v81 <= 0x40 )
                return 0;
LABEL_42:
              if ( v80 )
                j_j___libc_free_0_0(v80);
LABEL_44:
              if ( v26 )
                return sub_340F900(*a1, 196, a15, v67, v68, v35, v69, v65, a8);
              return 0;
            }
            v39 = *(_QWORD *)(v63 + 40);
            v57 = v78;
            v41 = *(_QWORD *)(v39 + 40);
            if ( v78 )
            {
              v42 = *(_DWORD *)(v39 + 48);
              goto LABEL_40;
            }
LABEL_110:
            if ( !v41 )
              goto LABEL_41;
LABEL_111:
            if ( !(unsigned __int8)sub_32657E0((__int64)&v80, *(_QWORD *)v39) )
              goto LABEL_41;
            goto LABEL_112;
          }
LABEL_56:
          if ( v74 )
            j_j___libc_free_0_0(v74);
          goto LABEL_58;
        }
        v55 = v78;
        v51 = *(_QWORD *)(v63 + 40);
        v53 = *(_QWORD *)(v51 + 40);
        if ( v78 )
        {
          v54 = *(_DWORD *)(v51 + 48);
          goto LABEL_75;
        }
LABEL_98:
        if ( !v53 )
          goto LABEL_76;
        goto LABEL_99;
      }
LABEL_49:
      if ( v74 )
        j_j___libc_free_0_0(v74);
      goto LABEL_51;
    }
    v46 = *(_QWORD *)(v16 + 40);
    v50 = v78;
    v48 = *(_QWORD *)(v46 + 40);
    if ( v78 )
    {
      v49 = *(_DWORD *)(v46 + 48);
      goto LABEL_64;
    }
LABEL_84:
    if ( !v48 )
      goto LABEL_65;
    goto LABEL_85;
  }
  v27 = a14;
  if ( a6 )
    v27 = a13;
  v28 = *a1;
  v29 = &a8;
  a14 = v27;
  if ( a6 )
    v29 = &a7;
  *((_QWORD *)&v56 + 1) = *((_QWORD *)&v64 + 1);
  *(_QWORD *)&v56 = a2;
  return sub_340F900(v28, a14, a15, v67, v68, v25, v56, v65, *v29);
}
