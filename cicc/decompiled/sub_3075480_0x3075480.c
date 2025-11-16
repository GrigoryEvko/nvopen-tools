// Function: sub_3075480
// Address: 0x3075480
//
__int64 __fastcall sub_3075480(
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
  __int64 v11; // r14
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
  __int64 v25; // rbx
  __int64 v26; // rcx
  __int64 v27; // rdx
  __int64 v28; // r12
  unsigned __int16 i; // r15
  unsigned __int16 v30; // bx
  __int64 v31; // r15
  unsigned int v32; // r12d
  int v33; // esi
  unsigned __int8 v34; // cl
  int v35; // ecx
  unsigned __int8 v36; // di
  signed __int64 v37; // rax
  unsigned __int64 v38; // rdx
  unsigned __int64 v39; // kr00_8
  __int64 v40; // rax
  __int64 v41; // r8
  __int64 v42; // r9
  signed __int64 v43; // r12
  __int64 *v44; // r9
  __int64 *v45; // rax
  __int64 v46; // r14
  unsigned int v47; // edx
  unsigned __int64 v48; // r12
  signed __int64 v49; // rax
  bool v50; // of
  unsigned __int64 v51; // r12
  __int64 v52; // rax
  unsigned __int64 v53; // kr10_8
  unsigned int v54; // ecx
  __int64 v55; // rdi
  __int64 v56; // r12
  __int64 v57; // rcx
  unsigned __int64 v58; // rax
  __int64 v59; // rsi
  __int64 v60; // rsi
  __int64 v61; // rsi
  __int64 *v62; // rax
  _BYTE *v63; // rdx
  __int64 v64; // rcx
  _BYTE *v65; // rdx
  _BYTE *v66; // rdx
  __int64 v67; // rcx
  __int64 v68; // rcx
  __int64 v69; // rax
  _BOOL8 v70; // rdx
  unsigned __int64 v71; // kr20_8
  signed __int64 v75; // [rsp+40h] [rbp-80h]
  unsigned int v76; // [rsp+48h] [rbp-78h]
  __int64 *v77; // [rsp+48h] [rbp-78h]
  __int64 v78; // [rsp+48h] [rbp-78h]
  __int64 v79; // [rsp+48h] [rbp-78h]
  unsigned __int64 v80; // [rsp+50h] [rbp-70h] BYREF
  __int64 v81; // [rsp+58h] [rbp-68h]
  _QWORD v82[12]; // [rsp+60h] [rbp-60h] BYREF

  v11 = *(_QWORD *)(a1 + 24);
  v76 = sub_2FEBEF0(v11, a2);
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
          v59 = *((_QWORD *)v17 - 4);
          if ( v59 )
          {
            if ( !*(_BYTE *)v59
              && *(_QWORD *)(v59 + 24) == *((_QWORD *)v17 + 10)
              && (*(_BYTE *)(v59 + 33) & 0x20) != 0
              && *(_DWORD *)(v59 + 36) == 169 )
            {
              ++v12;
              break;
            }
          }
        }
        v18 = v12[2];
        if ( *v18 == 85 )
        {
          v60 = *((_QWORD *)v18 - 4);
          if ( v60 )
          {
            if ( !*(_BYTE *)v60
              && *(_QWORD *)(v60 + 24) == *((_QWORD *)v18 + 10)
              && (*(_BYTE *)(v60 + 33) & 0x20) != 0
              && *(_DWORD *)(v60 + 36) == 169 )
            {
              v12 += 2;
              break;
            }
          }
        }
        v19 = v12[3];
        if ( *v19 == 85 )
        {
          v61 = *((_QWORD *)v19 - 4);
          if ( v61 )
          {
            if ( !*(_BYTE *)v61
              && *(_QWORD *)(v61 + 24) == *((_QWORD *)v19 + 10)
              && (*(_BYTE *)(v61 + 33) & 0x20) != 0
              && *(_DWORD *)(v61 + 36) == 169 )
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
          goto LABEL_108;
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
LABEL_108:
    if ( v15 != 2 )
    {
      if ( v15 != 3 )
      {
        if ( v15 != 1 )
          goto LABEL_20;
        goto LABEL_111;
      }
      v65 = *v12;
      if ( **v12 == 85 )
      {
        v68 = *((_QWORD *)v65 - 4);
        if ( v68 )
        {
          if ( !*(_BYTE *)v68
            && *(_QWORD *)(v68 + 24) == *((_QWORD *)v65 + 10)
            && (*(_BYTE *)(v68 + 33) & 0x20) != 0
            && *(_DWORD *)(v68 + 36) == 169 )
          {
            goto LABEL_16;
          }
        }
      }
      ++v12;
    }
    v66 = *v12;
    if ( **v12 == 85 )
    {
      v67 = *((_QWORD *)v66 - 4);
      if ( v67 )
      {
        if ( !*(_BYTE *)v67
          && *(_QWORD *)(v67 + 24) == *((_QWORD *)v66 + 10)
          && (*(_BYTE *)(v67 + 33) & 0x20) != 0
          && *(_DWORD *)(v67 + 36) == 169 )
        {
          goto LABEL_16;
        }
      }
    }
    ++v12;
LABEL_111:
    v63 = *v12;
    if ( **v12 != 85 )
      goto LABEL_20;
    v64 = *((_QWORD *)v63 - 4);
    if ( !v64
      || *(_BYTE *)v64
      || *(_QWORD *)(v64 + 24) != *((_QWORD *)v63 + 10)
      || (*(_BYTE *)(v64 + 33) & 0x20) == 0
      || *(_DWORD *)(v64 + 36) != 169 )
    {
      goto LABEL_20;
    }
    goto LABEL_16;
  }
  v75 = 1;
  v25 = *(_QWORD *)a3;
  v26 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), (__int64 *)a3, 0);
  v28 = v27;
  for ( i = v26; ; i = v81 )
  {
    LOWORD(v26) = i;
    sub_2FE6CC0((__int64)&v80, *(_QWORD *)(a1 + 24), v25, v26, v28);
    if ( (_BYTE)v80 == 10 )
      break;
    if ( !(_BYTE)v80 )
      goto LABEL_32;
    if ( (v80 & 0xFB) == 2 )
    {
      v52 = 2 * v75;
      if ( !is_mul_ok(2u, v75) )
      {
        v52 = 0x7FFFFFFFFFFFFFFFLL;
        if ( v75 <= 0 )
          v52 = 0x8000000000000000LL;
      }
      v75 = v52;
    }
    if ( (_WORD)v81 == i && ((_WORD)v81 || v82[0] == v28) )
    {
LABEL_32:
      v30 = i;
      v31 = a1;
      v32 = a2;
      goto LABEL_33;
    }
    v26 = v81;
    v28 = v82[0];
  }
  v30 = i;
  v31 = a1;
  v75 = 0;
  v32 = a2;
  if ( !v30 )
    v30 = 8;
LABEL_33:
  v33 = *(unsigned __int8 *)(a3 + 8);
  v34 = *(_BYTE *)(a3 + 8);
  if ( (unsigned int)(v33 - 17) <= 1 )
    v34 = *(_BYTE *)(**(_QWORD **)(a3 + 16) + 8LL);
  v22 = 2;
  if ( v34 > 3u && v34 != 5 )
    v22 = ((v34 & 0xFD) == 4) + 1LL;
  v35 = 1;
  if ( v30 == 1 || v30 && (v35 = v30, *(_QWORD *)(v11 + 8LL * v30 + 112)) )
  {
    if ( v76 > 0x1F3 )
    {
      if ( *(_QWORD *)(v11 + 8LL * v35 + 112) )
        goto LABEL_45;
    }
    else
    {
      v36 = *(_BYTE *)(v76 + v11 + 500LL * (unsigned int)v35 + 6414);
      if ( v36 <= 1u )
      {
        v53 = v22;
        v38 = v75 * v22;
        if ( !is_mul_ok(v75, v53) )
        {
          v38 = 0x8000000000000000LL;
          if ( v75 > 0 )
            return 0x7FFFFFFFFFFFFFFFLL;
        }
        return v38;
      }
      if ( *(_QWORD *)(v11 + 8LL * v35 + 112) && v36 != 2 )
      {
LABEL_45:
        v37 = 2 * v75;
        if ( is_mul_ok(2u, v75) )
        {
          v39 = v22;
          v38 = v37 * v22;
          if ( is_mul_ok(v37, v39) )
            return v38;
          if ( v37 <= 0 )
            return 0x8000000000000000LL;
        }
        else
        {
          if ( v75 <= 0 )
          {
            v71 = v22;
            v38 = v22 << 63;
            if ( is_mul_ok(0x8000000000000000LL, v71) )
              return v38;
            return 0x8000000000000000LL;
          }
          v69 = 0x7FFFFFFFFFFFFFFFLL * v22;
          v70 = ((unsigned __int64)v22 * (unsigned __int128)0x7FFFFFFFFFFFFFFFuLL) >> 64 != 0;
          if ( v69 >= 0 && !v70 )
            return v69;
        }
        return 0x7FFFFFFFFFFFFFFFLL;
      }
    }
  }
  if ( v76 - 61 > 1 )
  {
LABEL_53:
    if ( (_BYTE)v33 == 18 )
      return 0;
    if ( (_BYTE)v33 != 17 )
      return v22;
    v40 = sub_3075ED0(v31, v32, **(_QWORD **)(a3 + 16), 0, a5, a6, (__int64)a7, a8, a9);
    v80 = (unsigned __int64)v82;
    v43 = v40;
    v81 = 0x600000000LL;
    if ( a8 > 6 )
    {
      sub_C8D5F0((__int64)&v80, v82, a8, 8u, v41, v42);
      v62 = (__int64 *)v80;
      v44 = (__int64 *)(v80 + 8 * a8);
      if ( (__int64 *)v80 == v44 )
        goto LABEL_60;
      do
        *v62++ = a3;
      while ( v44 != v62 );
    }
    else
    {
      v44 = v82;
      if ( !a8 )
      {
LABEL_60:
        LODWORD(v81) = a8;
        v46 = v43 * *(unsigned int *)(a3 + 32);
        if ( !is_mul_ok(v43, *(unsigned int *)(a3 + 32)) )
        {
          if ( v43 <= 0 || (v46 = 0x7FFFFFFFFFFFFFFFLL, !*(_DWORD *)(a3 + 32)) )
            v46 = 0x8000000000000000LL;
        }
        v77 = v44;
        v48 = sub_30727B0(v31, a3, 1, 0);
        if ( a8 )
        {
          v49 = sub_30729F0(v31, a7, a8, v77, v47, (__int64)v77);
          v50 = __OFADD__(v49, v48);
          v51 = v49 + v48;
          if ( !v50 )
            goto LABEL_63;
        }
        else
        {
          v49 = sub_30727B0(v31, a3, 0, 1);
          v50 = __OFADD__(v49, v48);
          v51 = v49 + v48;
          if ( !v50 )
          {
LABEL_63:
            result = v46 + v51;
            if ( __OFADD__(v46, v51) )
            {
              result = 0x7FFFFFFFFFFFFFFFLL;
              if ( v46 <= 0 )
                result = 0x8000000000000000LL;
            }
            if ( (_QWORD *)v80 != v82 )
            {
              v78 = result;
              _libc_free(v80);
              return v78;
            }
            return result;
          }
        }
        v51 = 0x7FFFFFFFFFFFFFFFLL;
        if ( v49 <= 0 )
          v51 = 0x8000000000000000LL;
        goto LABEL_63;
      }
      v45 = v82;
      do
        *v45++ = a3;
      while ( &v82[a8] != v45 );
    }
    v44 = (__int64 *)v80;
    goto LABEL_60;
  }
  v54 = 1;
  if ( v30 != 1 )
  {
    if ( !v30 )
      goto LABEL_53;
    v54 = v30;
    if ( !*(_QWORD *)(v11 + 8LL * v30 + 112) )
    {
      v55 = 60;
      if ( v76 == 61 )
        goto LABEL_53;
LABEL_74:
      if ( !*(_QWORD *)(v11 + 8LL * (int)v54 + 112) )
        goto LABEL_53;
      goto LABEL_75;
    }
  }
  if ( (*(_BYTE *)((unsigned int)(v76 != 61) + 65 + v11 + 500LL * v54 + 6414) & 0xFB) == 0 )
    goto LABEL_76;
  v55 = (unsigned int)(v76 != 61) + 59;
  if ( v30 != 1 )
    goto LABEL_74;
LABEL_75:
  if ( (*(_BYTE *)(v55 + 500LL * v54 + v11 + 6414) & 0xFB) != 0 )
    goto LABEL_53;
LABEL_76:
  v79 = sub_3075ED0(v31, (unsigned int)(v76 == 61) + 19, a3, 0, a5, a6, 0, 0, 0);
  v56 = sub_3075ED0(v31, 17, a3, 0, 0, 0, 0, 0, 0);
  v57 = sub_3075ED0(v31, 15, a3, 0, 0, 0, 0, 0, 0);
  v58 = v56 + v79;
  if ( __OFADD__(v56, v79) )
  {
    v58 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v56 <= 0 )
      v58 = 0x8000000000000000LL;
  }
  v50 = __OFADD__(v57, v58);
  result = v57 + v58;
  if ( v50 )
  {
    result = 0x7FFFFFFFFFFFFFFFLL;
    if ( v57 <= 0 )
      return 0x8000000000000000LL;
  }
  return result;
}
