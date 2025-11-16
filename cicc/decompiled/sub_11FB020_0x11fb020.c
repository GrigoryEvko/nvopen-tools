// Function: sub_11FB020
// Address: 0x11fb020
//
__int64 __fastcall sub_11FB020(__int64 a1, _BYTE *a2, __int64 a3, unsigned int a4, char a5, char a6)
{
  unsigned int v9; // r15d
  _BYTE *v12; // rsi
  unsigned int v13; // edx
  unsigned int v14; // esi
  int v15; // eax
  unsigned int v16; // eax
  unsigned int v17; // r12d
  bool v18; // al
  _BYTE *v19; // rax
  unsigned int v20; // esi
  unsigned int v21; // edx
  int v23; // eax
  __int64 v24; // r8
  unsigned int v25; // eax
  __int64 v26; // r8
  bool v27; // al
  int v28; // eax
  bool v29; // al
  unsigned __int64 v31; // rax
  unsigned int v32; // ecx
  int v33; // eax
  unsigned int v34; // eax
  unsigned int v35; // eax
  unsigned int v36; // edx
  __int64 v37; // rax
  int v38; // ecx
  unsigned int v39; // ecx
  const void *v40; // rcx
  __int64 v41; // r12
  __int64 v42; // rdi
  unsigned int v43; // eax
  unsigned int v44; // eax
  unsigned __int64 v45; // r8
  int v46; // r15d
  int v47; // eax
  __int64 v48; // rax
  int v49; // esi
  unsigned __int64 v50; // rcx
  int v52; // r15d
  int v53; // eax
  unsigned __int64 v54; // r8
  unsigned __int64 v55; // rax
  int v56; // edx
  unsigned __int64 v57; // rdx
  unsigned int v59; // [rsp+Ch] [rbp-B4h]
  unsigned int v60; // [rsp+10h] [rbp-B0h]
  unsigned int v61; // [rsp+10h] [rbp-B0h]
  unsigned int v62; // [rsp+10h] [rbp-B0h]
  unsigned int v63; // [rsp+10h] [rbp-B0h]
  unsigned int v64; // [rsp+18h] [rbp-A8h]
  const void *v65; // [rsp+18h] [rbp-A8h]
  __int64 v66; // [rsp+18h] [rbp-A8h]
  const void *v67; // [rsp+18h] [rbp-A8h]
  unsigned int v68; // [rsp+18h] [rbp-A8h]
  __int64 v69; // [rsp+18h] [rbp-A8h]
  unsigned int v70; // [rsp+20h] [rbp-A0h]
  __int64 v71; // [rsp+20h] [rbp-A0h]
  unsigned int v72; // [rsp+20h] [rbp-A0h]
  unsigned int v73; // [rsp+20h] [rbp-A0h]
  unsigned int v74; // [rsp+20h] [rbp-A0h]
  unsigned int v75; // [rsp+28h] [rbp-98h]
  char v76; // [rsp+2Fh] [rbp-91h]
  unsigned __int64 v77; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v78; // [rsp+38h] [rbp-88h]
  __int64 v79; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v80; // [rsp+48h] [rbp-78h]
  __int64 v81; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v82; // [rsp+58h] [rbp-68h]
  _BYTE *v83; // [rsp+60h] [rbp-60h]
  int v84; // [rsp+68h] [rbp-58h]
  const void *v85; // [rsp+70h] [rbp-50h] BYREF
  unsigned int v86; // [rsp+78h] [rbp-48h]
  __int64 v87; // [rsp+80h] [rbp-40h] BYREF
  unsigned int v88; // [rsp+88h] [rbp-38h]

  if ( a4 - 32 <= 1 )
    goto LABEL_2;
  v9 = a4;
  v12 = (_BYTE *)(a3 + 24);
  if ( *(_BYTE *)a3 != 17 )
  {
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a3 + 8) + 8LL) - 17 > 1 )
      goto LABEL_2;
    if ( *(_BYTE *)a3 > 0x15u )
      goto LABEL_2;
    v19 = sub_AD7630(a3, 1, a3);
    if ( !v19 || *v19 != 17 )
      goto LABEL_2;
    v12 = v19 + 24;
  }
  v76 = 0;
  if ( (v9 & 0xFFFFFFFA) == 0x22 )
  {
    v76 = 1;
    v9 = sub_B52870(v9);
    v78 = *((_DWORD *)v12 + 2);
    if ( v78 <= 0x40 )
      goto LABEL_7;
  }
  else
  {
    v78 = *((_DWORD *)v12 + 2);
    if ( v78 <= 0x40 )
    {
LABEL_7:
      v77 = *(_QWORD *)v12;
      goto LABEL_8;
    }
  }
  sub_C43780((__int64)&v77, (const void **)v12);
LABEL_8:
  if ( ((v9 - 37) & 0xFFFFFFFB) != 0 )
    goto LABEL_9;
  v27 = sub_B532B0(v9);
  v13 = v78;
  if ( !v27 )
  {
    if ( v78 )
    {
      if ( v78 <= 0x40 )
      {
        v29 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v78) == v77;
      }
      else
      {
        v73 = v78;
        v28 = sub_C445E0((__int64)&v77);
        v13 = v73;
        v29 = v73 == v28;
      }
LABEL_75:
      if ( v29 )
        goto LABEL_76;
      goto LABEL_86;
    }
LABEL_2:
    *(_BYTE *)(a1 + 48) = 0;
    return a1;
  }
  v32 = v78 - 1;
  if ( v78 <= 0x40 )
  {
    v29 = (1LL << v32) - 1 == v77;
    goto LABEL_75;
  }
  if ( (*(_QWORD *)(v77 + 8LL * (v32 >> 6)) & (1LL << v32)) == 0 )
  {
    v63 = v78;
    v68 = v78 - 1;
    v33 = sub_C445E0((__int64)&v77);
    v13 = v63;
    if ( v33 == v68 )
      goto LABEL_76;
  }
LABEL_86:
  sub_C46250((__int64)&v77);
  v9 = sub_B53110(v9);
LABEL_9:
  v86 = 1;
  v85 = 0;
  v88 = 1;
  v87 = 0;
  if ( v9 != 36 )
  {
    if ( v9 != 40 )
      BUG();
    v20 = v78;
    v21 = v78;
    v71 = 1LL << ((unsigned __int8)v78 - 1);
    if ( v78 <= 0x40 )
    {
      v45 = v77;
      if ( v77 )
      {
        v81 = 0;
        goto LABEL_123;
      }
      v82 = v78;
      v81 = 1LL << ((unsigned __int8)v78 - 1);
    }
    else
    {
      v59 = v78;
      v75 = v78 - 1;
      v61 = v78;
      if ( v61 != (unsigned int)sub_C444A0((__int64)&v77) )
      {
        v82 = v61;
        sub_C43690((__int64)&v81, 0, 0);
        v20 = v82;
        if ( v82 > 0x40 )
        {
          *(_QWORD *)(v81 + 8LL * (v75 >> 6)) |= v71;
          v20 = v82;
          if ( v82 > 0x40 )
          {
            sub_C43C10(&v81, (__int64 *)&v77);
            v20 = v82;
            _R8 = v81;
            v80 = v82;
            v79 = v81;
            if ( v82 > 0x40 )
            {
              v62 = v82;
              v66 = v81;
              v23 = sub_C44630((__int64)&v79);
              v24 = v66;
              if ( v23 == 1 )
              {
                sub_C43780((__int64)&v81, (const void **)&v79);
                v20 = v82;
                if ( v82 > 0x40 )
                {
                  sub_C43D10((__int64)&v81);
                  goto LABEL_56;
                }
                _R8 = v81;
LABEL_150:
                v54 = ~_R8;
                v55 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v20;
                if ( !v20 )
                  v55 = 0;
                v81 = v55 & v54;
LABEL_56:
                sub_C46250((__int64)&v81);
                v25 = v82;
                v82 = 0;
                if ( v86 > 0x40 && v85 )
                {
                  v67 = (const void *)v81;
                  v72 = v25;
                  j_j___libc_free_0_0(v85);
                  v85 = v67;
                  v86 = v72;
                  if ( v82 > 0x40 && v81 )
                    j_j___libc_free_0_0(v81);
                }
                else
                {
                  v85 = (const void *)v81;
                  v86 = v25;
                }
                v82 = v78;
                v26 = 1LL << ((unsigned __int8)v78 - 1);
                if ( v78 > 0x40 )
                {
                  v74 = v78 - 1;
                  v69 = 1LL << ((unsigned __int8)v78 - 1);
                  sub_C43690((__int64)&v81, 0, 0);
                  v26 = v69;
                  if ( v82 > 0x40 )
                  {
                    *(_QWORD *)(v81 + 8LL * (v74 >> 6)) |= v69;
LABEL_64:
                    if ( v88 > 0x40 && v87 )
                      j_j___libc_free_0_0(v87);
                    v84 = 32;
                    v87 = v81;
                    v88 = v82;
                    goto LABEL_68;
                  }
                }
                else
                {
                  v81 = 0;
                }
                v81 |= v26;
                goto LABEL_64;
              }
              if ( (*(_QWORD *)(v66 + 8LL * ((v62 - 1) >> 6)) & (1LL << ((unsigned __int8)v62 - 1))) != 0 )
              {
                v52 = sub_C44500((__int64)&v79);
                v53 = sub_C44590((__int64)&v79);
                v24 = v66;
                if ( v53 + v52 == v62 )
                  goto LABEL_141;
              }
              *(_BYTE *)(a1 + 48) = 0;
              if ( v24 )
                j_j___libc_free_0_0(v24);
              goto LABEL_127;
            }
LABEL_125:
            if ( _R8 )
            {
              if ( (_R8 & (_R8 - 1)) == 0 )
              {
                v82 = v20;
                goto LABEL_150;
              }
              if ( _bittest64((const __int64 *)&_R8, ((_BYTE)v20 - 1) & 0x3F) )
              {
                if ( !v20 )
                  goto LABEL_162;
                v56 = 64;
                if ( _R8 << (64 - (unsigned __int8)v20) != -1 )
                {
                  _BitScanReverse64(&v57, ~(_R8 << (64 - (unsigned __int8)v20)));
                  v56 = v57 ^ 0x3F;
                }
                __asm { tzcnt   rax, r8 }
                if ( (unsigned int)_RAX > v20 )
                  LODWORD(_RAX) = v20;
                if ( v20 == (_DWORD)_RAX + v56 )
                {
LABEL_162:
                  if ( v86 <= 0x40 )
                  {
                    v85 = (const void *)_R8;
                    v86 = v20;
LABEL_142:
                    if ( v88 <= 0x40 && v78 <= 0x40 )
                    {
                      v88 = v78;
                      v87 = v77;
                    }
                    else
                    {
                      sub_C43990((__int64)&v87, (__int64)&v77);
                    }
                    v84 = 33;
LABEL_68:
                    if ( v80 > 0x40 && v79 )
                      j_j___libc_free_0_0(v79);
                    goto LABEL_25;
                  }
LABEL_141:
                  sub_C43990((__int64)&v85, (__int64)&v79);
                  goto LABEL_142;
                }
              }
            }
            *(_BYTE *)(a1 + 48) = 0;
LABEL_127:
            v17 = v88;
LABEL_30:
            if ( v17 > 0x40 && v87 )
              j_j___libc_free_0_0(v87);
            if ( v86 > 0x40 && v85 )
              j_j___libc_free_0_0(v85);
            v13 = v78;
            goto LABEL_37;
          }
          v48 = v81;
          v45 = v77;
LABEL_124:
          _R8 = v48 ^ v45;
          v80 = v20;
          v79 = _R8;
          goto LABEL_125;
        }
        v45 = v77;
LABEL_123:
        v48 = v81 | v71;
        goto LABEL_124;
      }
      v82 = v59;
      sub_C43690((__int64)&v81, 0, 0);
      if ( v82 <= 0x40 )
      {
        v34 = v86;
        v81 |= v71;
      }
      else
      {
        *(_QWORD *)(v81 + 8LL * (v75 >> 6)) |= v71;
        v34 = v86;
      }
      if ( v34 > 0x40 && v85 )
        j_j___libc_free_0_0(v85);
      v21 = v78;
    }
    v85 = (const void *)v81;
    v35 = v82;
    v82 = v21;
    v86 = v35;
    if ( v21 > 0x40 )
      sub_C43690((__int64)&v81, 0, 0);
    else
      v81 = 0;
    if ( v88 > 0x40 && v87 )
      j_j___libc_free_0_0(v87);
    v84 = 33;
    v87 = v81;
    v88 = v82;
    goto LABEL_25;
  }
  v13 = v78;
  v14 = v78;
  if ( v78 <= 0x40 )
  {
    _RAX = v77;
    if ( v77 )
    {
      if ( (v77 & (v77 - 1)) == 0 )
      {
        v82 = v78;
        goto LABEL_80;
      }
      if ( _bittest64((const __int64 *)&_RAX, ((_BYTE)v78 - 1) & 0x3F) )
      {
        if ( !v78 )
          goto LABEL_137;
        v49 = 64;
        if ( v77 << (64 - (unsigned __int8)v78) != -1 )
        {
          _BitScanReverse64(&v50, ~(v77 << (64 - (unsigned __int8)v78)));
          v49 = v50 ^ 0x3F;
        }
        __asm { tzcnt   rcx, rax }
        if ( (unsigned int)_RCX > v78 )
          LODWORD(_RCX) = v78;
        if ( v49 + (_DWORD)_RCX == v78 )
        {
LABEL_137:
          v85 = (const void *)v77;
          v86 = v78;
          goto LABEL_138;
        }
      }
    }
  }
  else
  {
    v60 = v78;
    v64 = v78;
    v15 = sub_C44630((__int64)&v77);
    v13 = v60;
    if ( v15 == 1 )
    {
      v82 = v64;
      sub_C43780((__int64)&v81, (const void **)&v77);
      v14 = v82;
      if ( v82 > 0x40 )
      {
        sub_C43D10((__int64)&v81);
LABEL_14:
        sub_C46250((__int64)&v81);
        v16 = v82;
        v82 = 0;
        if ( v86 > 0x40 && v85 )
        {
          v65 = (const void *)v81;
          v70 = v16;
          j_j___libc_free_0_0(v85);
          v85 = v65;
          v86 = v70;
          if ( v82 > 0x40 && v81 )
            j_j___libc_free_0_0(v81);
        }
        else
        {
          v85 = (const void *)v81;
          v86 = v16;
        }
        v82 = v78;
        if ( v78 > 0x40 )
          sub_C43690((__int64)&v81, 0, 0);
        else
          v81 = 0;
        if ( v88 > 0x40 && v87 )
          j_j___libc_free_0_0(v87);
        v84 = 32;
        v87 = v81;
        v88 = v82;
        goto LABEL_25;
      }
      _RAX = v81;
LABEL_80:
      v31 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v14) & ~_RAX;
      if ( !v14 )
        v31 = 0;
      v81 = v31;
      goto LABEL_14;
    }
    if ( (*(_QWORD *)(v77 + 8LL * ((v60 - 1) >> 6)) & (1LL << ((unsigned __int8)v60 - 1))) != 0 )
    {
      v46 = sub_C44500((__int64)&v77);
      v47 = sub_C44590((__int64)&v77);
      v13 = v60;
      if ( v47 + v46 == v60 )
      {
        sub_C43990((__int64)&v85, (__int64)&v77);
        if ( v88 > 0x40 || (v13 = v78, v78 > 0x40) )
        {
          sub_C43990((__int64)&v87, (__int64)&v77);
LABEL_121:
          v84 = 33;
LABEL_25:
          if ( a6
            || ((v17 = v88, v88 <= 0x40) ? (v18 = v87 == 0) : (v18 = v17 == (unsigned int)sub_C444A0((__int64)&v87)), v18) )
          {
            if ( v76 )
              v84 = sub_B52870(v84);
            if ( a5 && *a2 == 67 && (v41 = *((_QWORD *)a2 - 4)) != 0 )
            {
              v42 = *(_QWORD *)(v41 + 8);
              v83 = (_BYTE *)*((_QWORD *)a2 - 4);
              v43 = sub_BCB060(v42);
              sub_C449B0((__int64)&v81, &v85, v43);
              if ( v86 > 0x40 && v85 )
                j_j___libc_free_0_0(v85);
              v85 = (const void *)v81;
              v86 = v82;
              v44 = sub_BCB060(*(_QWORD *)(v41 + 8));
              sub_C449B0((__int64)&v81, (const void **)&v87, v44);
              if ( v88 > 0x40 && v87 )
                j_j___libc_free_0_0(v87);
              v37 = v81;
              v36 = v82;
              a2 = v83;
            }
            else
            {
              v36 = v88;
              v37 = v87;
            }
            v38 = v84;
            *(_DWORD *)(a1 + 40) = v36;
            *(_QWORD *)a1 = a2;
            v13 = v78;
            *(_DWORD *)(a1 + 8) = v38;
            v39 = v86;
            *(_QWORD *)(a1 + 32) = v37;
            *(_DWORD *)(a1 + 24) = v39;
            v40 = v85;
            *(_BYTE *)(a1 + 48) = 1;
            *(_QWORD *)(a1 + 16) = v40;
            goto LABEL_37;
          }
          *(_BYTE *)(a1 + 48) = 0;
          goto LABEL_30;
        }
LABEL_138:
        v88 = v13;
        v87 = v77;
        goto LABEL_121;
      }
    }
  }
LABEL_76:
  *(_BYTE *)(a1 + 48) = 0;
LABEL_37:
  if ( v13 > 0x40 && v77 )
    j_j___libc_free_0_0(v77);
  return a1;
}
