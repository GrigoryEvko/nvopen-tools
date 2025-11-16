// Function: sub_34D2250
// Address: 0x34d2250
//
__int64 __fastcall sub_34D2250(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        int a4,
        int a5,
        int a6,
        _BYTE **a7,
        unsigned __int64 a8,
        __int64 a9)
{
  __int64 v11; // rbx
  _BYTE **v12; // rax
  _BYTE **v13; // rdi
  __int64 v14; // rcx
  __int64 v15; // rdx
  _BYTE **v16; // rcx
  _BYTE *v17; // rdx
  _BYTE *v18; // rdx
  _BYTE *v19; // rdx
  _BYTE *v20; // rdx
  __int64 v21; // rsi
  __int64 v22; // rdx
  __int64 result; // rax
  int v24; // eax
  __int64 v25; // rcx
  __int64 v26; // rbx
  __int64 v27; // rdx
  __int64 v28; // r12
  unsigned __int16 i; // r15
  unsigned __int16 v30; // si
  __int64 v31; // rbx
  __int64 v32; // r14
  __int64 v33; // r12
  unsigned int v34; // r13d
  int v35; // ecx
  unsigned __int8 v36; // al
  int v37; // eax
  unsigned __int8 v38; // di
  signed __int64 v39; // rax
  unsigned __int64 v40; // rdx
  unsigned __int64 v41; // kr00_8
  signed __int64 v42; // rbx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 *v45; // r11
  __int64 *v46; // rax
  __int64 v47; // r13
  unsigned __int64 v48; // rbx
  signed __int64 v49; // rax
  bool v50; // of
  unsigned __int64 v51; // rbx
  __int64 v52; // rax
  unsigned __int64 v53; // kr10_8
  __int64 v54; // rax
  __int64 v55; // rdi
  __int64 v56; // rbx
  __int64 v57; // r15
  __int64 v58; // rcx
  unsigned __int64 v59; // rax
  __int64 v60; // rsi
  __int64 v61; // rsi
  __int64 v62; // rsi
  __int64 *v63; // rax
  _BYTE *v64; // rdx
  __int64 v65; // rcx
  _BYTE *v66; // rdx
  _BYTE *v67; // rdx
  __int64 v68; // rcx
  __int64 v69; // rcx
  __int64 v70; // rax
  _BOOL8 v71; // rdx
  unsigned __int64 v72; // kr20_8
  __int64 v73; // [rsp+8h] [rbp-B8h]
  __int64 v74; // [rsp+10h] [rbp-B0h]
  __int64 v75; // [rsp+18h] [rbp-A8h]
  __int64 *v79; // [rsp+28h] [rbp-98h]
  signed __int64 v81; // [rsp+40h] [rbp-80h]
  unsigned int v82; // [rsp+48h] [rbp-78h]
  __int64 v83; // [rsp+48h] [rbp-78h]
  unsigned __int64 v84; // [rsp+50h] [rbp-70h] BYREF
  __int64 v85; // [rsp+58h] [rbp-68h]
  _QWORD v86[12]; // [rsp+60h] [rbp-60h] BYREF

  v11 = *(_QWORD *)(a1 + 24);
  v82 = sub_2FEBEF0(v11, a2);
  if ( a4 )
  {
    if ( a2 <= 0x18 )
    {
      v22 = 4;
      if ( a2 > 0x12 )
        return v22;
      goto LABEL_20;
    }
    if ( a2 - 28 > 1 )
      goto LABEL_20;
    v12 = a7;
    v13 = &a7[a8];
    v14 = (__int64)(8 * a8) >> 5;
    v15 = (__int64)(8 * a8) >> 3;
    if ( v14 > 0 )
    {
      v16 = &a7[4 * v14];
      while ( 1 )
      {
        v20 = *v12;
        if ( **v12 == 85 )
        {
          v21 = *((_QWORD *)v20 - 4);
          if ( v21 )
          {
            if ( !*(_BYTE *)v21
              && *(_QWORD *)(v21 + 24) == *((_QWORD *)v20 + 10)
              && (*(_BYTE *)(v21 + 33) & 0x20) != 0
              && *(_DWORD *)(v21 + 36) == 169 )
            {
              break;
            }
          }
        }
        v17 = v12[1];
        if ( *v17 == 85 )
        {
          v60 = *((_QWORD *)v17 - 4);
          if ( v60 )
          {
            if ( !*(_BYTE *)v60
              && *(_QWORD *)(v60 + 24) == *((_QWORD *)v17 + 10)
              && (*(_BYTE *)(v60 + 33) & 0x20) != 0
              && *(_DWORD *)(v60 + 36) == 169 )
            {
              ++v12;
              break;
            }
          }
        }
        v18 = v12[2];
        if ( *v18 == 85 )
        {
          v61 = *((_QWORD *)v18 - 4);
          if ( v61 )
          {
            if ( !*(_BYTE *)v61
              && *(_QWORD *)(v61 + 24) == *((_QWORD *)v18 + 10)
              && (*(_BYTE *)(v61 + 33) & 0x20) != 0
              && *(_DWORD *)(v61 + 36) == 169 )
            {
              v12 += 2;
              break;
            }
          }
        }
        v19 = v12[3];
        if ( *v19 == 85 )
        {
          v62 = *((_QWORD *)v19 - 4);
          if ( v62 )
          {
            if ( !*(_BYTE *)v62
              && *(_QWORD *)(v62 + 24) == *((_QWORD *)v19 + 10)
              && (*(_BYTE *)(v62 + 33) & 0x20) != 0
              && *(_DWORD *)(v62 + 36) == 169 )
            {
              v12 += 3;
              break;
            }
          }
        }
        v12 += 4;
        if ( v16 == v12 )
        {
          v15 = v13 - v12;
          goto LABEL_107;
        }
      }
LABEL_16:
      v22 = 0;
      if ( v13 != v12 )
        return v22;
LABEL_20:
      v22 = 1;
      if ( a4 == 1 )
      {
        v24 = *(unsigned __int8 *)(a3 + 8);
        if ( (unsigned int)(v24 - 17) <= 1 )
          LOBYTE(v24) = *(_BYTE *)(**(_QWORD **)(a3 + 16) + 8LL);
        v22 = 3;
        if ( (unsigned __int8)v24 > 3u && (_BYTE)v24 != 5 )
          return 2LL * ((v24 & 0xFD) == 4) + 1;
      }
      return v22;
    }
LABEL_107:
    if ( v15 != 2 )
    {
      if ( v15 != 3 )
      {
        if ( v15 != 1 )
          goto LABEL_20;
        goto LABEL_110;
      }
      v66 = *v12;
      if ( **v12 == 85 )
      {
        v69 = *((_QWORD *)v66 - 4);
        if ( v69 )
        {
          if ( !*(_BYTE *)v69
            && *(_QWORD *)(v69 + 24) == *((_QWORD *)v66 + 10)
            && (*(_BYTE *)(v69 + 33) & 0x20) != 0
            && *(_DWORD *)(v69 + 36) == 169 )
          {
            goto LABEL_16;
          }
        }
      }
      ++v12;
    }
    v67 = *v12;
    if ( **v12 == 85 )
    {
      v68 = *((_QWORD *)v67 - 4);
      if ( v68 )
      {
        if ( !*(_BYTE *)v68
          && *(_QWORD *)(v68 + 24) == *((_QWORD *)v67 + 10)
          && (*(_BYTE *)(v68 + 33) & 0x20) != 0
          && *(_DWORD *)(v68 + 36) == 169 )
        {
          goto LABEL_16;
        }
      }
    }
    ++v12;
LABEL_110:
    v64 = *v12;
    if ( **v12 != 85 )
      goto LABEL_20;
    v65 = *((_QWORD *)v64 - 4);
    if ( !v65
      || *(_BYTE *)v65
      || *(_QWORD *)(v65 + 24) != *((_QWORD *)v64 + 10)
      || (*(_BYTE *)(v65 + 33) & 0x20) == 0
      || *(_DWORD *)(v65 + 36) != 169 )
    {
      goto LABEL_20;
    }
    goto LABEL_16;
  }
  v75 = *(_QWORD *)a3;
  v81 = 1;
  v74 = v11;
  v25 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), (__int64 *)a3, 0);
  v26 = v75;
  v73 = a3;
  v28 = v27;
  for ( i = v25; ; i = v85 )
  {
    LOWORD(v25) = i;
    sub_2FE6CC0((__int64)&v84, *(_QWORD *)(a1 + 24), v26, v25, v28);
    v30 = v85;
    if ( (_BYTE)v84 == 10 )
      break;
    if ( !(_BYTE)v84 )
    {
      v31 = v74;
      v32 = a1;
      v33 = v73;
      v34 = a2;
      v30 = i;
      goto LABEL_33;
    }
    if ( (v84 & 0xFB) == 2 )
    {
      v52 = 2 * v81;
      if ( !is_mul_ok(2u, v81) )
      {
        v52 = 0x7FFFFFFFFFFFFFFFLL;
        if ( v81 <= 0 )
          v52 = 0x8000000000000000LL;
      }
      v81 = v52;
    }
    if ( i == (_WORD)v85 && ((_WORD)v85 || v86[0] == v28) )
    {
      v31 = v74;
      v32 = a1;
      v33 = v73;
      v34 = a2;
      goto LABEL_33;
    }
    v25 = v85;
    v28 = v86[0];
  }
  v30 = 8;
  v32 = a1;
  v81 = 0;
  v31 = v74;
  v33 = v73;
  v34 = a2;
  if ( i )
    v30 = i;
LABEL_33:
  v35 = *(unsigned __int8 *)(v33 + 8);
  v36 = *(_BYTE *)(v33 + 8);
  if ( (unsigned int)(v35 - 17) <= 1 )
    v36 = *(_BYTE *)(**(_QWORD **)(v33 + 16) + 8LL);
  v22 = 2;
  if ( v36 > 3u && v36 != 5 )
    v22 = ((v36 & 0xFD) == 4) + 1LL;
  v37 = 1;
  if ( v30 == 1 || v30 && (v37 = v30, *(_QWORD *)(v31 + 8LL * v30 + 112)) )
  {
    if ( v82 > 0x1F3 )
    {
      if ( *(_QWORD *)(v31 + 8LL * v37 + 112) )
        goto LABEL_45;
    }
    else
    {
      v38 = *(_BYTE *)(v82 + v31 + 500LL * (unsigned int)v37 + 6414);
      if ( v38 <= 1u )
      {
        v53 = v22;
        v40 = v81 * v22;
        if ( !is_mul_ok(v81, v53) )
        {
          v40 = 0x8000000000000000LL;
          if ( v81 > 0 )
            return 0x7FFFFFFFFFFFFFFFLL;
        }
        return v40;
      }
      if ( *(_QWORD *)(v31 + 8LL * v37 + 112) && v38 != 2 )
      {
LABEL_45:
        v39 = 2 * v81;
        if ( is_mul_ok(2u, v81) )
        {
          v41 = v22;
          v40 = v39 * v22;
          if ( is_mul_ok(v39, v41) )
            return v40;
          if ( v39 <= 0 )
            return 0x8000000000000000LL;
        }
        else
        {
          if ( v81 <= 0 )
          {
            v72 = v22;
            v40 = v22 << 63;
            if ( is_mul_ok(0x8000000000000000LL, v72) )
              return v40;
            return 0x8000000000000000LL;
          }
          v70 = 0x7FFFFFFFFFFFFFFFLL * v22;
          v71 = ((unsigned __int64)v22 * (unsigned __int128)0x7FFFFFFFFFFFFFFFuLL) >> 64 != 0;
          if ( v70 >= 0 && !v71 )
            return v70;
        }
        return 0x7FFFFFFFFFFFFFFFLL;
      }
    }
  }
  if ( v82 - 61 > 1 )
  {
LABEL_53:
    if ( (_BYTE)v35 == 18 )
      return 0;
    if ( (_BYTE)v35 != 17 )
      return v22;
    v42 = sub_34D2250(v32, v34, **(_QWORD **)(v33 + 16), 0, a5, a6, (__int64)a7, a8, a9);
    v84 = (unsigned __int64)v86;
    v85 = 0x600000000LL;
    if ( a8 > 6 )
    {
      sub_C8D5F0((__int64)&v84, v86, a8, 8u, v43, v44);
      v63 = (__int64 *)v84;
      v45 = (__int64 *)(v84 + 8 * a8);
      if ( (__int64 *)v84 == v45 )
      {
LABEL_60:
        LODWORD(v85) = a8;
        v47 = v42 * *(unsigned int *)(v33 + 32);
        if ( !is_mul_ok(v42, *(unsigned int *)(v33 + 32)) )
        {
          if ( v42 <= 0 || (v47 = 0x7FFFFFFFFFFFFFFFLL, !*(_DWORD *)(v33 + 32)) )
            v47 = 0x8000000000000000LL;
        }
        v79 = v45;
        v48 = sub_34D2080(v32, v33, 1, 0);
        if ( a8 )
          v49 = sub_34D0EB0(v32, a7, a8, v79, (unsigned int)a8, 0);
        else
          v49 = sub_34D2080(v32, v33, 0, 1);
        v50 = __OFADD__(v49, v48);
        v51 = v49 + v48;
        if ( v50 )
        {
          v51 = 0x7FFFFFFFFFFFFFFFLL;
          if ( v49 <= 0 )
            v51 = 0x8000000000000000LL;
        }
        result = v47 + v51;
        if ( __OFADD__(v47, v51) )
        {
          result = 0x7FFFFFFFFFFFFFFFLL;
          if ( v47 <= 0 )
            result = 0x8000000000000000LL;
        }
        if ( (_QWORD *)v84 != v86 )
        {
          v83 = result;
          _libc_free(v84);
          return v83;
        }
        return result;
      }
      do
        *v63++ = v33;
      while ( v45 != v63 );
    }
    else
    {
      v45 = v86;
      if ( !a8 )
        goto LABEL_60;
      v46 = v86;
      do
        *v46++ = v33;
      while ( &v86[a8] != v46 );
    }
    v45 = (__int64 *)v84;
    goto LABEL_60;
  }
  v54 = 1;
  if ( v30 != 1 )
  {
    if ( !v30 )
      goto LABEL_53;
    v54 = v30;
    if ( !*(_QWORD *)(v31 + 8LL * v30 + 112) )
    {
      v55 = 60;
      if ( v82 == 61 )
        goto LABEL_53;
LABEL_123:
      if ( !*(_QWORD *)(v31 + 8LL * (int)v54 + 112) )
        goto LABEL_53;
      goto LABEL_76;
    }
  }
  if ( (*(_BYTE *)((unsigned int)(v82 != 61) + 65 + v31 + 500LL * (unsigned int)v54 + 6414) & 0xFB) == 0 )
    goto LABEL_77;
  v55 = (unsigned int)(v82 != 61) + 59;
  if ( v30 != 1 )
    goto LABEL_123;
LABEL_76:
  if ( (*(_BYTE *)(v55 + 500 * v54 + v31 + 6414) & 0xFB) != 0 )
    goto LABEL_53;
LABEL_77:
  v56 = sub_34D2250(v32, (unsigned int)(v82 == 61) + 19, v33, 0, a5, a6, 0, 0, 0);
  v57 = sub_34D2250(v32, 17, v33, 0, 0, 0, 0, 0, 0);
  v58 = sub_34D2250(v32, 15, v33, 0, 0, 0, 0, 0, 0);
  v59 = v57 + v56;
  if ( __OFADD__(v57, v56) )
  {
    v59 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v57 <= 0 )
      v59 = 0x8000000000000000LL;
  }
  v50 = __OFADD__(v58, v59);
  result = v58 + v59;
  if ( v50 )
  {
    result = 0x7FFFFFFFFFFFFFFFLL;
    if ( v58 <= 0 )
      return 0x8000000000000000LL;
  }
  return result;
}
