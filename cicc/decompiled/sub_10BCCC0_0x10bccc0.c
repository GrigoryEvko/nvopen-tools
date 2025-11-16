// Function: sub_10BCCC0
// Address: 0x10bccc0
//
__int64 __fastcall sub_10BCCC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6, unsigned int **a7)
{
  bool v7; // bl
  __int64 v8; // r13
  unsigned int v10; // r14d
  __int64 v11; // rdx
  int v13; // r15d
  bool v15; // al
  unsigned int v16; // r15d
  __int64 v17; // rax
  unsigned int v18; // r14d
  __int64 v19; // rax
  __int64 v21; // r14
  __int64 v22; // rdx
  _BYTE *v23; // rax
  unsigned int v24; // r15d
  __int64 v25; // rsi
  int v27; // ecx
  unsigned __int64 v28; // rax
  unsigned __int64 v30; // r15
  __int64 v31; // r14
  __int64 v32; // rdx
  _BYTE *v33; // rax
  unsigned int v34; // r15d
  __int64 v35; // r14
  int v36; // eax
  __int64 v37; // r14
  __int64 v38; // rdx
  _BYTE *v39; // rax
  unsigned __int8 *v40; // r15
  unsigned int v41; // ebx
  __int64 v42; // r15
  __int64 v43; // rax
  int v44; // r14d
  int v45; // r14d
  bool v46; // al
  int v47; // ebx
  int v48; // ebx
  char v49; // r15
  unsigned int v50; // r14d
  __int64 v51; // rax
  unsigned int v52; // esi
  __int64 v54; // r8
  int v55; // ecx
  unsigned __int64 v56; // rax
  char v58; // r15
  unsigned int v59; // r14d
  __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // r14
  int v63; // r14d
  unsigned int v64; // r15d
  __int64 v65; // rax
  __int64 v66; // rax
  __int64 v67; // r15
  int v68; // r14d
  int v69; // r14d
  __int64 v70; // r15
  __int64 v71; // r14
  int v72; // ebx
  int v73; // ebx
  int v74; // r15d
  int v75; // r15d
  int v76; // ebx
  int v77; // ebx
  unsigned int v78; // [rsp+Ch] [rbp-84h]
  int v79; // [rsp+Ch] [rbp-84h]
  unsigned __int8 *v81; // [rsp+18h] [rbp-78h]
  int v82; // [rsp+18h] [rbp-78h]
  __int64 v83; // [rsp+18h] [rbp-78h]
  int v84; // [rsp+18h] [rbp-78h]
  __int64 v85; // [rsp+20h] [rbp-70h]
  int v86; // [rsp+20h] [rbp-70h]
  int v87; // [rsp+20h] [rbp-70h]
  int v88; // [rsp+20h] [rbp-70h]
  int v89; // [rsp+20h] [rbp-70h]
  __int64 v90; // [rsp+20h] [rbp-70h]
  _BYTE v92[32]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v93; // [rsp+50h] [rbp-40h]

  v7 = a6 != 33 || a5 != 32;
  if ( v7 )
    return 0;
  v8 = a2;
  if ( *(_BYTE *)a2 == 17 )
  {
    v10 = *(_DWORD *)(a2 + 32);
    v11 = 1LL << ((unsigned __int8)v10 - 1);
    _RAX = *(_QWORD *)(a2 + 24);
    if ( v10 <= 0x40 )
    {
      if ( (v11 & _RAX) == 0 )
        return 0;
      if ( v10 )
      {
        v13 = 64;
        if ( _RAX << (64 - (unsigned __int8)v10) != -1 )
        {
          _BitScanReverse64(&v30, ~(_RAX << (64 - (unsigned __int8)v10)));
          v13 = v30 ^ 0x3F;
        }
      }
      else
      {
        v13 = 0;
      }
      __asm { tzcnt   rax, rax }
      if ( (unsigned int)_RAX > v10 )
        LODWORD(_RAX) = *(_DWORD *)(a2 + 32);
    }
    else
    {
      if ( (*(_QWORD *)(_RAX + 8LL * ((v10 - 1) >> 6)) & v11) == 0 )
        return 0;
      v13 = sub_C44500(a2 + 24);
      LODWORD(_RAX) = sub_C44590(a2 + 24);
    }
    v15 = v13 + (_DWORD)_RAX == v10;
  }
  else
  {
    v21 = *(_QWORD *)(a2 + 8);
    v22 = (unsigned int)*(unsigned __int8 *)(v21 + 8) - 17;
    if ( (unsigned int)v22 > 1 || *(_BYTE *)a2 > 0x15u )
      return 0;
    v23 = sub_AD7630(a2, 0, v22);
    if ( !v23 || *v23 != 17 )
    {
      if ( *(_BYTE *)(v21 + 8) == 17 )
      {
        v86 = *(_DWORD *)(v21 + 32);
        if ( v86 )
        {
          v49 = 0;
          v50 = 0;
          while ( 1 )
          {
            v51 = sub_AD69F0((unsigned __int8 *)v8, v50);
            if ( !v51 )
              break;
            if ( *(_BYTE *)v51 != 13 )
            {
              if ( *(_BYTE *)v51 != 17 )
                return 0;
              v52 = *(_DWORD *)(v51 + 32);
              _RDI = *(_QWORD *)(v51 + 24);
              v54 = 1LL << ((unsigned __int8)v52 - 1);
              if ( v52 > 0x40 )
              {
                if ( (*(_QWORD *)(_RDI + 8LL * ((v52 - 1) >> 6)) & v54) == 0 )
                  return 0;
                v70 = v51 + 24;
                v78 = *(_DWORD *)(v51 + 32);
                v82 = sub_C44500(v51 + 24);
                LODWORD(_RAX) = sub_C44590(v70);
                v52 = v78;
                v55 = v82;
              }
              else
              {
                if ( (v54 & _RDI) == 0 )
                  return 0;
                if ( v52 )
                {
                  v55 = 64;
                  if ( _RDI << (64 - (unsigned __int8)v52) != -1 )
                  {
                    _BitScanReverse64(&v56, ~(_RDI << (64 - (unsigned __int8)v52)));
                    v55 = v56 ^ 0x3F;
                  }
                }
                else
                {
                  v55 = 0;
                }
                __asm { tzcnt   rax, rdi }
                if ( (unsigned int)_RAX > v52 )
                  LODWORD(_RAX) = v52;
              }
              if ( v52 != v55 + (_DWORD)_RAX )
                return 0;
              v49 = 1;
            }
            if ( v86 == ++v50 )
            {
              if ( v49 )
                goto LABEL_8;
              return 0;
            }
          }
        }
      }
      return 0;
    }
    v24 = *((_DWORD *)v23 + 8);
    v25 = 1LL << ((unsigned __int8)v24 - 1);
    _RDX = *((_QWORD *)v23 + 3);
    if ( v24 > 0x40 )
    {
      if ( (*(_QWORD *)(_RDX + 8LL * ((v24 - 1) >> 6)) & v25) == 0 )
        return 0;
      v62 = (__int64)(v23 + 24);
      v88 = sub_C44500((__int64)(v23 + 24));
      LODWORD(_RAX) = sub_C44590(v62);
      v27 = v88;
    }
    else
    {
      if ( (v25 & _RDX) == 0 )
        return 0;
      if ( v24 )
      {
        v27 = 64;
        if ( _RDX << (64 - (unsigned __int8)v24) != -1 )
        {
          _BitScanReverse64(&v28, ~(_RDX << (64 - (unsigned __int8)v24)));
          v27 = v28 ^ 0x3F;
        }
      }
      else
      {
        v27 = 0;
      }
      __asm { tzcnt   rax, rdx }
      if ( (unsigned int)_RAX > v24 )
        LODWORD(_RAX) = v24;
    }
    v15 = v27 + (_DWORD)_RAX == v24;
  }
  if ( !v15 )
    return 0;
LABEL_8:
  if ( *(_BYTE *)a3 == 17 )
  {
    v16 = *(_DWORD *)(a3 + 32);
    if ( v16 <= 0x40 )
    {
      v17 = *(_QWORD *)(a3 + 24);
      if ( !v17 )
        return 0;
      goto LABEL_11;
    }
    v44 = sub_C44630(a3 + 24);
    v45 = sub_C444A0(a3 + 24) + v44;
    v46 = (unsigned int)sub_C44590(a3 + 24) + v45 == v16;
  }
  else
  {
    v37 = *(_QWORD *)(a3 + 8);
    v38 = (unsigned int)*(unsigned __int8 *)(v37 + 8) - 17;
    if ( (unsigned int)v38 > 1 || *(_BYTE *)a3 > 0x15u )
      return 0;
    v39 = sub_AD7630(a3, 0, v38);
    if ( !v39 || *v39 != 17 )
    {
      if ( *(_BYTE *)(v37 + 8) == 17 )
      {
        v87 = *(_DWORD *)(v37 + 32);
        if ( v87 )
        {
          v58 = 0;
          v59 = 0;
          while ( 1 )
          {
            v60 = sub_AD69F0((unsigned __int8 *)a3, v59);
            if ( !v60 )
              break;
            if ( *(_BYTE *)v60 != 13 )
            {
              if ( *(_BYTE *)v60 != 17 )
                return 0;
              if ( *(_DWORD *)(v60 + 32) > 0x40u )
              {
                v79 = *(_DWORD *)(v60 + 32);
                v83 = v60 + 24;
                v74 = sub_C44630(v60 + 24);
                v75 = sub_C444A0(v83) + v74;
                if ( v79 != (unsigned int)sub_C44590(v83) + v75 )
                  return 0;
              }
              else
              {
                v61 = *(_QWORD *)(v60 + 24);
                if ( !v61 || (((v61 - 1) | v61) & (((v61 - 1) | v61) + 1)) != 0 )
                  return 0;
              }
              v58 = 1;
            }
            if ( v87 == ++v59 )
            {
              if ( v58 )
                goto LABEL_12;
              return 0;
            }
          }
        }
      }
      return 0;
    }
    if ( *((_DWORD *)v39 + 8) <= 0x40u )
    {
      v17 = *((_QWORD *)v39 + 3);
      if ( !v17 )
        return 0;
LABEL_11:
      if ( (((v17 - 1) | v17) & (((v17 - 1) | v17) + 1)) == 0 )
        goto LABEL_12;
      return 0;
    }
    v67 = (__int64)(v39 + 24);
    v89 = *((_DWORD *)v39 + 8);
    v68 = sub_C44630((__int64)(v39 + 24));
    v69 = sub_C444A0(v67) + v68;
    v46 = (unsigned int)sub_C44590(v67) + v69 == v89;
  }
  if ( !v46 )
    return 0;
LABEL_12:
  if ( *(_BYTE *)a4 == 17 )
  {
    v18 = *(_DWORD *)(a4 + 32);
    if ( v18 > 0x40 )
    {
      v47 = sub_C44630(a4 + 24);
      v48 = sub_C444A0(a4 + 24) + v47;
      if ( v18 != (unsigned int)sub_C44590(a4 + 24) + v48 )
        return 0;
      goto LABEL_42;
    }
    v19 = *(_QWORD *)(a4 + 24);
    if ( !v19 )
      return 0;
  }
  else
  {
    v31 = *(_QWORD *)(a4 + 8);
    v32 = (unsigned int)*(unsigned __int8 *)(v31 + 8) - 17;
    if ( (unsigned int)v32 > 1 || *(_BYTE *)a4 > 0x15u )
      return 0;
    v33 = sub_AD7630(a4, 0, v32);
    if ( !v33 || *v33 != 17 )
    {
      if ( *(_BYTE *)(v31 + 8) == 17 )
      {
        v63 = *(_DWORD *)(v31 + 32);
        if ( v63 )
        {
          v64 = 0;
          while ( 1 )
          {
            v65 = sub_AD69F0((unsigned __int8 *)a4, v64);
            if ( !v65 )
              break;
            if ( *(_BYTE *)v65 != 13 )
            {
              if ( *(_BYTE *)v65 != 17 )
                return 0;
              if ( *(_DWORD *)(v65 + 32) > 0x40u )
              {
                v84 = *(_DWORD *)(v65 + 32);
                v90 = v65 + 24;
                v76 = sub_C44630(v65 + 24);
                v77 = sub_C444A0(v90) + v76;
                if ( v84 != (unsigned int)sub_C44590(v90) + v77 )
                  return 0;
              }
              else
              {
                v66 = *(_QWORD *)(v65 + 24);
                if ( !v66 || (((v66 - 1) | v66) & (((v66 - 1) | v66) + 1)) != 0 )
                  return 0;
              }
              v7 = 1;
            }
            if ( v63 == ++v64 )
            {
              if ( v7 )
                goto LABEL_42;
              return 0;
            }
          }
        }
      }
      return 0;
    }
    v34 = *((_DWORD *)v33 + 8);
    if ( v34 > 0x40 )
    {
      v71 = (__int64)(v33 + 24);
      v72 = sub_C44630((__int64)(v33 + 24));
      v73 = sub_C444A0(v71) + v72;
      if ( v34 != (unsigned int)sub_C44590(v71) + v73 )
        return 0;
      goto LABEL_42;
    }
    v19 = *((_QWORD *)v33 + 3);
    if ( !v19 )
      return 0;
  }
  if ( (((v19 - 1) | v19) & (((v19 - 1) | v19) + 1)) != 0 )
    return 0;
LABEL_42:
  v35 = *(_QWORD *)(v8 + 8);
  v36 = *(unsigned __int8 *)(v35 + 8);
  if ( v36 == 17 )
  {
LABEL_54:
    v40 = 0;
    if ( *(_BYTE *)v8 >= 0x16u )
      v8 = 0;
    if ( *(_BYTE *)a3 <= 0x15u )
      v40 = (unsigned __int8 *)a3;
    v81 = v40;
    if ( *(_BYTE *)a4 > 0x15u || !v35 || !v8 || !v40 )
      return 0;
    v41 = 0;
    if ( *(_DWORD *)(v35 + 32) )
    {
      while ( 1 )
      {
        v42 = sub_AD69F0((unsigned __int8 *)v8, v41);
        v85 = sub_AD69F0(v81, v41);
        v43 = sub_AD69F0((unsigned __int8 *)a4, v41);
        if ( v85 == 0 || v42 == 0 || !v43 || !sub_10B8D70(v42, v85, v43) )
          return 0;
        if ( ++v41 == *(_DWORD *)(v35 + 32) )
          goto LABEL_45;
      }
    }
    goto LABEL_45;
  }
  if ( v36 == 18 )
  {
    v35 = 0;
    goto LABEL_54;
  }
  if ( !sub_10B8D70(v8, a3, a4) )
    return 0;
LABEL_45:
  v93 = 257;
  return sub_92B530(a7, 0x24u, a1, (_BYTE *)a3, (__int64)v92);
}
