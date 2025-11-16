// Function: sub_15823D0
// Address: 0x15823d0
//
__int64 __fastcall sub_15823D0(__int64 a1, __int64 a2, char a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 v6; // al
  __int64 v7; // r12
  __int64 v8; // r13
  unsigned __int8 v9; // bl
  unsigned __int8 v10; // dl
  unsigned int v12; // r13d
  unsigned int k; // r14d
  __int64 v14; // rcx
  unsigned int v15; // r14d
  unsigned int v16; // eax
  char v17; // al
  unsigned int v18; // edx
  _BYTE *v19; // r14
  unsigned __int16 v20; // ax
  unsigned __int8 v21; // al
  unsigned __int8 v22; // al
  __int64 v23; // rax
  __int64 v24; // rbx
  __int64 v25; // rax
  unsigned int v26; // r15d
  bool v27; // al
  unsigned __int8 v28; // al
  __int64 v29; // rax
  __int64 v30; // r8
  __int64 v31; // r9
  unsigned int v32; // r14d
  __int64 v34; // rax
  __int64 v35; // r8
  __int64 v36; // r9
  unsigned int v37; // r15d
  __int64 v38; // rdi
  __int64 v39; // rax
  unsigned int v40; // ebx
  int v41; // eax
  char v42; // r12
  __int64 v43; // rax
  _BYTE *v44; // r15
  unsigned int v45; // r14d
  __int64 v46; // r14
  __int64 v47; // rax
  int v48; // edx
  __int64 v49; // r9
  _QWORD *v50; // rax
  int v51; // ebx
  __int64 v52; // r14
  unsigned int v53; // eax
  __int64 v54; // rax
  unsigned __int64 v55; // r9
  unsigned __int64 v56; // r15
  __int64 v57; // r10
  __int64 v58; // rcx
  unsigned int v59; // r8d
  unsigned __int64 v60; // rax
  int v61; // eax
  unsigned int v62; // r11d
  unsigned __int64 v63; // rax
  int v64; // eax
  __int64 *v65; // rax
  __int64 v66; // r8
  __int64 *v67; // rax
  __int64 v68; // rcx
  __int64 v69; // rcx
  __int64 v70; // r8
  char v71; // al
  __int64 v72; // rax
  __int64 v73; // rax
  int v74; // esi
  unsigned int v75; // esi
  __int64 j; // r12
  __int64 v77; // rax
  int v78; // [rsp+4h] [rbp-6Ch]
  unsigned int v79; // [rsp+8h] [rbp-68h]
  int v80; // [rsp+8h] [rbp-68h]
  unsigned int v81; // [rsp+Ch] [rbp-64h]
  unsigned int v82; // [rsp+Ch] [rbp-64h]
  __int64 v83; // [rsp+10h] [rbp-60h]
  __int64 v84; // [rsp+10h] [rbp-60h]
  __int64 v85; // [rsp+18h] [rbp-58h]
  __int64 v86; // [rsp+18h] [rbp-58h]
  __int64 v87; // [rsp+20h] [rbp-50h]
  char v88; // [rsp+2Bh] [rbp-45h]
  unsigned __int8 v89; // [rsp+2Ch] [rbp-44h]
  _QWORD *i; // [rsp+30h] [rbp-40h]
  __int64 v91; // [rsp+38h] [rbp-38h]

  while ( 1 )
  {
    if ( a1 == a2 )
      return 32;
    v6 = *(_BYTE *)(a1 + 16);
    v7 = a1;
    v8 = a2;
    v9 = a3;
    if ( v6 != 5 )
      break;
    v18 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
    v14 = v18;
    v19 = *(_BYTE **)(a1 - 24LL * v18);
    v20 = *(_WORD *)(a1 + 18);
    if ( v20 > 0x26u )
    {
      if ( v20 <= 0x2Au )
      {
        if ( v20 <= 0x28u )
          return 42;
      }
      else if ( v20 != 47 )
      {
        return 42;
      }
    }
    else if ( v20 <= 0x24u )
    {
      if ( v20 != 32 )
        return 42;
      v21 = *(_BYTE *)(a2 + 16);
      if ( v21 != 15 )
      {
        if ( v21 > 3u )
        {
          if ( *(_WORD *)(a2 + 18) == 32 && v19[16] <= 3u )
          {
            v44 = *(_BYTE **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
            if ( v44[16] <= 3u )
            {
              if ( v19 == v44 )
              {
                if ( (unsigned __int8)((__int64 (*)(void))sub_1594530)() && (unsigned __int8)sub_1594530(a2) )
                {
                  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
                    v46 = *(_QWORD *)(a1 - 8);
                  else
                    v46 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
                  v47 = sub_16348C0(a1);
                  v48 = *(_DWORD *)(a2 + 20);
                  v89 = v9;
                  v49 = v47 | 4;
                  v50 = (_QWORD *)(v46 + 24);
                  v51 = *(_DWORD *)(a1 + 20);
                  v52 = 1;
                  for ( i = v50; ; i += 3 )
                  {
                    v53 = v51 & 0xFFFFFFF;
                    if ( (_DWORD)v52 == (v51 & 0xFFFFFFF) )
                    {
                      v9 = v89;
LABEL_149:
                      for ( j = v53; ; ++j )
                      {
                        v77 = v48 & 0xFFFFFFF;
                        if ( (unsigned int)v77 <= (unsigned int)j )
                          break;
                        if ( !(unsigned __int8)sub_1593BB0(*(_QWORD *)(v8 + 24 * (j - v77))) )
                        {
                          if ( *(_BYTE *)(*(_QWORD *)(v8 + 24 * (j - (*(_DWORD *)(v8 + 20) & 0xFFFFFFF))) + 16LL) != 13 )
                            return 42;
                          goto LABEL_124;
                        }
                        v48 = *(_DWORD *)(v8 + 20);
                      }
                      return 32;
                    }
                    if ( (_DWORD)v52 == (v48 & 0xFFFFFFF) )
                    {
                      v74 = v51;
                      v9 = v89;
                      while ( 1 )
                      {
                        v75 = v74 & 0xFFFFFFF;
                        v53 = v52;
                        if ( v75 <= (unsigned int)v52 )
                        {
                          v48 = *(_DWORD *)(v8 + 20);
                          goto LABEL_149;
                        }
                        if ( !(unsigned __int8)sub_1593BB0(*(_QWORD *)(a1 + 24 * (v52 - v75))) )
                          break;
                        v74 = *(_DWORD *)(a1 + 20);
                        ++v52;
                      }
                      if ( *(_BYTE *)(*(_QWORD *)(a1 + 24 * (v52 - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF))) + 16LL) != 13 )
                        return 42;
                      return v9 == 0 ? 34 : 38;
                    }
                    v54 = v49;
                    v55 = v49 & 0xFFFFFFFFFFFFFFF8LL;
                    v91 = v55;
                    v56 = v55;
                    v88 = (v54 >> 2) & 1;
                    if ( !v88 || (v87 = v55) == 0 )
                    {
                      v72 = sub_1643D30(v55, *i);
                      v51 = *(_DWORD *)(a1 + 20);
                      v48 = *(_DWORD *)(a2 + 20);
                      v87 = v72;
                    }
                    v57 = *(_QWORD *)(a2 + 24 * ((unsigned int)v52 - (unsigned __int64)(v48 & 0xFFFFFFF)));
                    v58 = *(_QWORD *)(a1 + 24 * ((unsigned int)v52 - (unsigned __int64)(v51 & 0xFFFFFFF)));
                    if ( v58 != v57 )
                    {
                      if ( *(_BYTE *)(v58 + 16) != 13 || *(_BYTE *)(v57 + 16) != 13 )
                        return 42;
                      v59 = *(_DWORD *)(v58 + 32);
                      if ( v59 > 0x40 )
                      {
                        v80 = v48;
                        v82 = *(_DWORD *)(v58 + 32);
                        v84 = *(_QWORD *)(a2 + 24 * ((unsigned int)v52 - (unsigned __int64)(v48 & 0xFFFFFFF)));
                        v86 = *(_QWORD *)(a1 + 24 * ((unsigned int)v52 - (unsigned __int64)(v51 & 0xFFFFFFF)));
                        v61 = sub_16A57B0(v58 + 24);
                        v48 = v80;
                        v59 = v82;
                        v57 = v84;
                        v58 = v86;
                      }
                      else
                      {
                        v60 = *(_QWORD *)(v58 + 24);
                        if ( v60 )
                        {
                          _BitScanReverse64(&v60, v60);
                          LODWORD(v60) = v60 ^ 0x3F;
                        }
                        else
                        {
                          LODWORD(v60) = 64;
                        }
                        v61 = v59 + v60 - 64;
                      }
                      if ( v59 - v61 > 0x40 )
                        return 42;
                      v62 = *(_DWORD *)(v57 + 32);
                      if ( v62 > 0x40 )
                      {
                        v78 = v48;
                        v79 = *(_DWORD *)(v57 + 32);
                        v81 = v59;
                        v83 = v58;
                        v85 = v57;
                        v64 = sub_16A57B0(v57 + 24);
                        v48 = v78;
                        v62 = v79;
                        v59 = v81;
                        v58 = v83;
                        v57 = v85;
                      }
                      else
                      {
                        v63 = *(_QWORD *)(v57 + 24);
                        if ( v63 )
                        {
                          _BitScanReverse64(&v63, v63);
                          LODWORD(v63) = v63 ^ 0x3F;
                        }
                        else
                        {
                          LODWORD(v63) = 64;
                        }
                        v64 = v62 + v63 - 64;
                      }
                      if ( v62 - v64 > 0x40 )
                        return 42;
                      v65 = *(__int64 **)(v58 + 24);
                      v66 = v59 > 0x40
                          ? *v65
                          : (__int64)((_QWORD)v65 << (64 - (unsigned __int8)v59)) >> (64 - (unsigned __int8)v59);
                      v67 = *(__int64 **)(v57 + 24);
                      v68 = v62 > 0x40
                          ? *v67
                          : (__int64)((_QWORD)v67 << (64 - (unsigned __int8)v62)) >> (64 - (unsigned __int8)v62);
                      if ( v68 != v66 )
                        break;
                    }
                    v52 = (unsigned int)(v52 + 1);
                    if ( !v88 || !v56 )
                    {
                      v73 = sub_1643D30(v56, *i);
                      v51 = *(_DWORD *)(a1 + 20);
                      v48 = *(_DWORD *)(a2 + 20);
                      v91 = v73;
                    }
                    v71 = *(_BYTE *)(v91 + 8);
                    if ( ((v71 - 14) & 0xFD) != 0 )
                    {
                      v49 = 0;
                      if ( v71 == 13 )
                        v49 = v91;
                    }
                    else
                    {
                      v49 = *(_QWORD *)(v91 + 24) | 4LL;
                    }
                  }
                  v9 = v89;
                  if ( (unsigned __int8)sub_1580C80(v87) )
                    return 42;
                  if ( v69 > v70 )
                  {
LABEL_124:
                    v45 = v9 == 0 ? 0xFFFFFFFC : 0;
                    return v45 + 40;
                  }
                  return v9 == 0 ? 34 : 38;
                }
              }
              else if ( (unsigned __int8)sub_1582340((__int64 *)a1) )
              {
                a2 = (__int64)v44;
                if ( (unsigned __int8)sub_1582340((__int64 *)v8) )
                  goto LABEL_87;
              }
            }
          }
        }
        else
        {
          v22 = v19[16];
          if ( v22 == 15 )
          {
            v45 = v9 == 0 ? 0xFFFFFFFC : 0;
            if ( (*(_BYTE *)(a2 + 32) & 0xF) == 9 )
              return v45 + 41;
            else
              return v45 + 40;
          }
          if ( v22 <= 3u )
          {
            if ( v19 != (_BYTE *)a2 )
            {
              v23 = 24LL * v18;
              if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
              {
                v24 = *(_QWORD *)(a1 - 8);
                v7 = v24 + v23;
              }
              else
              {
                v24 = a1 - v23;
              }
              while ( 1 )
              {
                v24 += 24;
                if ( v24 == v7 )
                  break;
                v25 = *(_QWORD *)v24;
                if ( *(_BYTE *)(*(_QWORD *)v24 + 16LL) != 13 )
                  return 42;
                v26 = *(_DWORD *)(v25 + 32);
                if ( v26 <= 0x40 )
                  v27 = *(_QWORD *)(v25 + 24) == 0;
                else
                  v27 = v26 == (unsigned int)sub_16A57B0(v25 + 24);
                if ( !v27 )
                  return 42;
              }
LABEL_87:
              a1 = (__int64)v19;
              return sub_1581270(a1, a2);
            }
            return v9 == 0 ? 34 : 38;
          }
        }
        return 42;
      }
      v28 = v19[16];
      if ( v28 <= 3u )
      {
        if ( (v19[32] & 0xF) != 9 )
          return v9 == 0 ? 34 : 38;
        return v9 == 0 ? 35 : 39;
      }
      if ( v28 != 15 )
        return 42;
      if ( v18 != 1 )
      {
        v12 = v18 - 1;
        for ( k = 1; (unsigned __int8)sub_1593BB0(*(_QWORD *)(a1 + 24 * (k - v14))); ++k )
        {
          if ( v12 == k )
            return 32;
          v14 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
        }
        return v9 == 0 ? 34 : 38;
      }
      return 32;
    }
    if ( (unsigned __int8)(*(_BYTE *)(*(_QWORD *)v19 + 8LL) - 1) <= 5u
      || !(unsigned __int8)sub_1593BB0(a2)
      || (*(_BYTE *)(*(_QWORD *)a1 + 8LL) & 0xFB) != 0xB )
    {
      return 42;
    }
    v41 = *(unsigned __int16 *)(a1 + 18);
    v42 = 0;
    if ( v41 != 37 )
    {
      v42 = 1;
      if ( v41 != 38 )
        v42 = v9;
    }
    v43 = sub_15A06D0(*(_QWORD *)v19);
    a3 = v42;
    a1 = (__int64)v19;
    a2 = v43;
  }
  if ( v6 > 4u )
  {
    if ( *(_BYTE *)(a2 + 16) <= 5u )
      goto LABEL_17;
    v29 = sub_15A35F0(32, a1, a2, 0, a5, a6);
    if ( *(_BYTE *)(v29 + 16) == 13 )
    {
      v32 = *(_DWORD *)(v29 + 32);
      if ( !(v32 <= 0x40 ? *(_QWORD *)(v29 + 24) == 0 : v32 == (unsigned int)sub_16A57B0(v29 + 24)) )
        return 32;
    }
    if ( v9 )
    {
      v34 = sub_15A35F0(40, a1, a2, 0, v30, v31);
      v15 = 40;
      if ( *(_BYTE *)(v34 + 16) != 13 )
        goto LABEL_75;
    }
    else
    {
      v34 = sub_15A35F0(36, a1, a2, 0, v30, v31);
      v15 = 36;
      if ( *(_BYTE *)(v34 + 16) != 13 )
      {
LABEL_62:
        v38 = 34;
        v15 = 34;
        goto LABEL_63;
      }
    }
    v37 = *(_DWORD *)(v34 + 32);
    if ( v37 <= 0x40 )
    {
      if ( *(_QWORD *)(v34 + 24) )
        return v15;
    }
    else if ( v37 != (unsigned int)sub_16A57B0(v34 + 24) )
    {
      return v15;
    }
    if ( !v9 )
      goto LABEL_62;
LABEL_75:
    v38 = 38;
    v15 = 38;
LABEL_63:
    v39 = sub_15A35F0(v38, v7, a2, 0, v35, v36);
    if ( *(_BYTE *)(v39 + 16) == 13 )
    {
      v40 = *(_DWORD *)(v39 + 32);
      if ( v40 <= 0x40 )
      {
        if ( *(_QWORD *)(v39 + 24) )
          return v15;
      }
      else if ( v40 != (unsigned int)sub_16A57B0(v39 + 24) )
      {
        return v15;
      }
    }
    return 42;
  }
  if ( v6 > 3u )
  {
    v17 = *(_BYTE *)(a2 + 16);
    if ( v17 != 5 )
    {
      if ( v17 == 4 && *(_QWORD *)(a2 - 48) == *(_QWORD *)(a1 - 48) )
        return 42;
      return 33;
    }
  }
  else
  {
    v10 = *(_BYTE *)(a2 + 16);
    if ( v10 != 5 )
    {
      if ( v10 <= 3u )
        return sub_1581270(a1, a2);
      if ( v10 != 4
        && ((*(_BYTE *)(a1 + 32) & 0xF) == 9
         || v6 == 1
         || (unsigned __int8)sub_15E4690(0, *(_DWORD *)(*(_QWORD *)a1 + 8LL) >> 8)) )
      {
        return 42;
      }
      return 33;
    }
  }
LABEL_17:
  v16 = sub_15823D0(a2, a1, v9);
  if ( v16 == 42 )
    return 42;
  return sub_15FF5D0(v16);
}
