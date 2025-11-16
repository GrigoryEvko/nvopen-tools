// Function: sub_E80970
// Address: 0xe80970
//
__int64 __fastcall sub_E80970(int *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned __int8 a6)
{
  __int64 v6; // rdi
  __int64 v7; // rbx
  __int64 v8; // rbp
  __int64 v9; // r12
  __int64 v10; // r13
  __int64 v11; // r14
  __int64 v16; // rdi
  __int64 v17; // r8
  unsigned int v18; // eax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // r15
  unsigned int v22; // r10d
  char v23; // al
  __int64 v24; // rdi
  char v25; // al
  unsigned int v26; // edx
  int v27; // r15d
  int v28; // eax
  unsigned __int16 v29; // r10
  unsigned __int64 v30; // rax
  __int64 v31; // rax
  __int64 v33; // rdi
  unsigned int v34; // r15d
  char v35; // al
  __int64 v36; // rcx
  _BYTE *v37; // rax
  _BYTE *v38; // rsi
  _BYTE *v39; // rdi
  unsigned int v40; // edx
  __int64 (*v41)(); // rdx
  unsigned int v42; // eax
  unsigned int v43; // esi
  __int64 v44; // rax
  __m128i v45; // xmm0
  int v46; // eax
  __int64 (*v47)(); // rdx
  __int64 v48; // rax
  bool v49; // zf
  void *v50; // rax
  __int16 v51; // [rsp-ACh] [rbp-ACh]
  __int64 v52; // [rsp-A8h] [rbp-A8h]
  unsigned __int8 v53; // [rsp-A8h] [rbp-A8h]
  unsigned __int16 v54; // [rsp-A0h] [rbp-A0h]
  unsigned __int8 v55; // [rsp-A0h] [rbp-A0h]
  __int64 v56; // [rsp-A0h] [rbp-A0h]
  __int64 v57; // [rsp-98h] [rbp-98h] BYREF
  __int64 v58; // [rsp-90h] [rbp-90h]
  __int64 v59; // [rsp-88h] [rbp-88h]
  int v60; // [rsp-80h] [rbp-80h]
  __int64 v61; // [rsp-78h] [rbp-78h] BYREF
  __int64 v62; // [rsp-70h] [rbp-70h]
  __int64 v63; // [rsp-68h] [rbp-68h]
  int v64; // [rsp-60h] [rbp-60h]
  __m128i v65; // [rsp-58h] [rbp-58h] BYREF
  __int64 v66; // [rsp-48h] [rbp-48h]
  int v67; // [rsp-40h] [rbp-40h]
  __int64 v68; // [rsp-30h] [rbp-30h]
  __int64 v69; // [rsp-28h] [rbp-28h]
  __int64 v70; // [rsp-20h] [rbp-20h]
  __int64 v71; // [rsp-18h] [rbp-18h]
  __int64 v72; // [rsp-8h] [rbp-8h]

  v72 = v8;
  v71 = v11;
  v70 = v10;
  v69 = v9;
  v68 = v7;
  switch ( *(_BYTE *)a1 )
  {
    case 0:
      v33 = *((_QWORD *)a1 + 2);
      v34 = a6;
      v52 = a4;
      v57 = 0;
      v58 = 0;
      v59 = 0;
      v60 = 0;
      v61 = 0;
      v62 = 0;
      v63 = 0;
      v64 = 0;
      v35 = sub_E80970(v33, &v57, a3, a4, a5, a6);
      v36 = v52;
      if ( !v35 || (v42 = sub_E80970(*((_QWORD *)a1 + 3), &v61, a3, v52, a5, v34), v17 = v42, !(_BYTE)v42) )
      {
        v37 = (_BYTE *)*((_QWORD *)a1 + 2);
        if ( *v37 != 4 )
          goto LABEL_52;
        v38 = (_BYTE *)*((_QWORD *)a1 + 3);
        if ( *v38 != 4 )
          goto LABEL_52;
        v39 = v37 - 8;
        v40 = (unsigned int)*a1 >> 8;
        if ( v40 == 3 )
        {
          v47 = *(__int64 (**)())(*((_QWORD *)v37 - 1) + 40LL);
          v31 = 0;
          if ( v47 != sub_E7FAA0 )
            v31 = -(__int64)((unsigned __int8 (__fastcall *)(_BYTE *, _BYTE *, __int64 (*)(), __int64, __int64))v47)(
                              v39,
                              v38,
                              v47,
                              v36,
                              v17);
        }
        else
        {
          if ( v40 != 12 )
            goto LABEL_52;
          v41 = *(__int64 (**)())(*((_QWORD *)v37 - 1) + 40LL);
          v31 = -1;
          if ( v41 != sub_E7FAA0 )
            v31 = -(__int64)(((unsigned __int8 (__fastcall *)(_BYTE *, _BYTE *, __int64 (*)(), __int64))v41)(
                               v39,
                               v38,
                               v41,
                               v36)
                           ^ 1u);
        }
LABEL_34:
        *(_QWORD *)a2 = 0;
        LODWORD(v17) = 1;
        *(_QWORD *)(a2 + 8) = 0;
        *(_QWORD *)(a2 + 16) = v31;
        *(_DWORD *)(a2 + 24) = 0;
        return (unsigned int)v17;
      }
      v43 = (unsigned int)*a1 >> 8;
      if ( !v57 && !v58 && !v61 && !v62 )
      {
        switch ( v43 )
        {
          case 0u:
            v6 = v63 + v59;
            goto LABEL_74;
          case 1u:
            v6 = v63 & v59;
            goto LABEL_74;
          case 2u:
          case 0xAu:
            if ( !v63 )
              goto LABEL_52;
            v6 = v59 / v63;
            if ( v43 == 2 )
              goto LABEL_74;
            v6 = v59 % v63;
            if ( ((1LL << v43) & 0x1338) == 0 )
              goto LABEL_74;
            goto LABEL_81;
          case 3u:
            v6 = v63 == v59;
            goto LABEL_81;
          case 4u:
            v6 = v63 < v59;
            goto LABEL_81;
          case 5u:
            v6 = v63 <= v59;
            goto LABEL_81;
          case 6u:
            v6 = (v59 != 0) & (unsigned __int8)(v63 != 0);
            goto LABEL_74;
          case 7u:
            v6 = (v59 | v63) != 0;
            goto LABEL_74;
          case 8u:
            v6 = v63 > v59;
            goto LABEL_81;
          case 9u:
            v6 = v63 >= v59;
            goto LABEL_81;
          case 0xBu:
            v6 = v63 * v59;
            goto LABEL_74;
          case 0xCu:
            v6 = v63 != v59;
LABEL_81:
            *(_QWORD *)a2 = 0;
            *(_QWORD *)(a2 + 8) = 0;
            *(_DWORD *)(a2 + 24) = 0;
            *(_QWORD *)(a2 + 16) = -(__int64)(v6 != 0);
            return (unsigned int)v17;
          case 0xDu:
            v6 = v63 | v59;
            goto LABEL_74;
          case 0xEu:
            v6 = ~v63 | v59;
            goto LABEL_74;
          case 0xFu:
            v6 = v59 << v63;
            goto LABEL_74;
          case 0x10u:
            v6 = v59 >> v63;
            goto LABEL_74;
          case 0x11u:
            v6 = (unsigned __int64)v59 >> v63;
            goto LABEL_74;
          case 0x12u:
            v6 = v59 - v63;
            goto LABEL_74;
          case 0x13u:
            v6 = v63 ^ v59;
            goto LABEL_74;
          default:
            v6 = 0;
LABEL_74:
            *(_QWORD *)a2 = 0;
            *(_QWORD *)(a2 + 8) = 0;
            *(_QWORD *)(a2 + 16) = v6;
            *(_DWORD *)(a2 + 24) = 0;
            break;
        }
        return (unsigned int)v17;
      }
      if ( !v43 )
      {
        v66 = v63;
        v65.m128i_i64[0] = v61;
        v44 = v62;
        goto LABEL_49;
      }
      if ( v43 != 18 )
      {
LABEL_52:
        LODWORD(v17) = 0;
        return (unsigned int)v17;
      }
      v66 = -v63;
      v65.m128i_i64[0] = v62;
      v44 = v61;
LABEL_49:
      v65.m128i_i64[1] = v44;
      v67 = v64;
      LODWORD(v17) = sub_E817B0(a3, a5, v34, &v57, &v65, a2);
      return (unsigned int)v17;
    case 1:
      v31 = *((_QWORD *)a1 + 2);
      goto LABEL_34;
    case 2:
      v21 = *((_QWORD *)a1 + 2);
      v22 = (unsigned int)*a1 >> 8;
      v23 = *(_BYTE *)(v21 + 9) & 0x70;
      if ( a3 && *(_BYTE *)(a3 + 32) )
      {
        if ( v23 != 32 )
          goto LABEL_51;
      }
      else if ( v23 != 32 || (_WORD)v22 )
      {
        goto LABEL_51;
      }
      if ( *(char *)(v21 + 8) < 0 )
        goto LABEL_51;
      v24 = *(_QWORD *)(v21 + 24);
      v25 = *(_BYTE *)(v21 + 8) | 8;
      *(_BYTE *)(v21 + 8) = v25;
      if ( *(_BYTE *)v24 == 2 && *(_WORD *)(v24 + 1) == 29 )
        goto LABEL_51;
      if ( a6 )
        goto LABEL_21;
      if ( *(_QWORD *)v21 )
        goto LABEL_19;
      if ( *(char *)(v21 + 8) >= 0 )
      {
        v51 = v22;
        v53 = a6;
        v56 = a4;
        v50 = sub_E807D0(v24);
        a4 = v56;
        a6 = v53;
        *(_QWORD *)v21 = v50;
        LOWORD(v22) = v51;
        if ( v50 )
        {
LABEL_19:
          if ( off_4C5D170 != *(_UNKNOWN **)v21 )
            goto LABEL_51;
        }
        v25 = *(_BYTE *)(v21 + 8);
        v24 = *(_QWORD *)(v21 + 24);
      }
LABEL_21:
      v26 = *a1;
      *(_BYTE *)(v21 + 8) = v25 | 8;
      v54 = v22;
      v27 = (v26 >> 8) & 0x10000;
      v28 = sub_E80970(v24, a2, a3, a4, a5, (unsigned __int8)((v27 != 0) | a6));
      v29 = v54;
      LODWORD(v17) = v28;
      if ( !(_BYTE)v28 )
        goto LABEL_51;
      if ( !v54 )
        goto LABEL_28;
      if ( *(_QWORD *)a2 )
      {
        if ( *(_DWORD *)(a2 + 24) )
          goto LABEL_52;
        v55 = v28;
        if ( *(_QWORD *)(a2 + 8) || *(_QWORD *)(a2 + 16) )
          goto LABEL_52;
        v30 = sub_E808D0(*(_QWORD *)(*(_QWORD *)a2 + 16LL), v29, *(_QWORD **)a3, 0);
        LODWORD(v17) = v55;
        *(_QWORD *)(a2 + 8) = 0;
        *(_QWORD *)a2 = v30;
        *(_QWORD *)(a2 + 16) = 0;
        *(_DWORD *)(a2 + 24) = 0;
LABEL_28:
        if ( !v27 )
          return (unsigned int)v17;
        if ( *(_OWORD *)a2 == 0 )
          return (unsigned int)v17;
        if ( !*(_QWORD *)(a2 + 16) )
        {
          LOBYTE(v17) = *(_QWORD *)(a2 + 8) == 0 || *(_QWORD *)a2 == 0;
          if ( (_BYTE)v17 )
            return (unsigned int)v17;
        }
      }
      else
      {
        LODWORD(v17) = 0;
        if ( *(_QWORD *)(a2 + 8) )
          return (unsigned int)v17;
      }
LABEL_51:
      *(_QWORD *)a2 = a1;
      LODWORD(v17) = 1;
      *(_QWORD *)(a2 + 8) = 0;
      *(_QWORD *)(a2 + 16) = 0;
      *(_DWORD *)(a2 + 24) = 0;
      return (unsigned int)v17;
    case 3:
      v16 = *((_QWORD *)a1 + 2);
      v65 = 0u;
      v66 = 0;
      v67 = 0;
      LODWORD(v17) = sub_E80970(v16, &v65, a3, a4, a5, a6);
      if ( !(_BYTE)v17 )
        return (unsigned int)v17;
      v18 = (unsigned int)*a1 >> 8;
      if ( v18 == 2 )
      {
        if ( *(_OWORD *)&v65 != 0 )
          goto LABEL_52;
        v48 = v66;
        *(_QWORD *)a2 = 0;
        *(_QWORD *)(a2 + 8) = 0;
        *(_DWORD *)(a2 + 24) = 0;
        *(_QWORD *)(a2 + 16) = ~v48;
      }
      else
      {
        if ( v18 > 2 )
        {
          if ( v18 == 3 )
          {
            v45 = _mm_loadu_si128(&v65);
            *(_QWORD *)(a2 + 16) = v66;
            v46 = v67;
            *(__m128i *)a2 = v45;
            *(_DWORD *)(a2 + 24) = v46;
          }
          return (unsigned int)v17;
        }
        if ( v18 )
        {
          v19 = v65.m128i_i64[0];
          if ( !v65.m128i_i64[0] || v65.m128i_i64[1] )
          {
            v20 = v66;
            *(_QWORD *)a2 = v65.m128i_i64[1];
            *(_QWORD *)(a2 + 8) = v19;
            *(_DWORD *)(a2 + 24) = 0;
            *(_QWORD *)(a2 + 16) = -v20;
            return (unsigned int)v17;
          }
          goto LABEL_52;
        }
        if ( *(_OWORD *)&v65 != 0 )
          goto LABEL_52;
        v49 = v66 == 0;
        *(_QWORD *)a2 = 0;
        *(_QWORD *)(a2 + 8) = 0;
        *(_QWORD *)(a2 + 16) = v49;
        *(_DWORD *)(a2 + 24) = 0;
      }
      return (unsigned int)v17;
    case 4:
      return (*(__int64 (__fastcall **)(int *, __int64, __int64))(*((_QWORD *)a1 - 1) + 32LL))(a1 - 2, a2, a3);
    default:
      BUG();
  }
}
