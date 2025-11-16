// Function: sub_9704C0
// Address: 0x9704c0
//
__int64 __fastcall sub_9704C0(unsigned __int8 *a1, __int64 a2, __int64 a3, unsigned int a4, _BYTE *a5)
{
  __int64 v5; // rdx
  unsigned __int8 *v6; // r15
  unsigned __int64 v7; // r12
  unsigned int v11; // r13d
  unsigned int v12; // r13d
  unsigned __int64 i; // r13
  unsigned __int64 v14; // rdi
  int v15; // ecx
  char v16; // al
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // r13
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  unsigned int v24; // eax
  __int64 v25; // r13
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  __int64 v28; // r14
  unsigned __int64 v29; // r12
  unsigned int j; // ebx
  char v31; // al
  __int64 v32; // rax
  int v33; // edi
  unsigned __int64 v34; // rdx
  int v35; // ebx
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // r13
  __int64 v39; // rax
  __int64 v40; // rsi
  __int64 v41; // r13
  char v42; // r14
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rax
  unsigned int v46; // r10d
  unsigned __int64 v47; // r14
  unsigned __int64 v48; // rax
  unsigned __int64 v49; // rdx
  __int64 v50; // rbx
  unsigned int v51; // r12d
  unsigned __int64 v52; // r13
  unsigned __int64 v53; // r14
  __int64 v54; // rax
  unsigned __int64 v55; // rax
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // rax
  char v59; // dl
  char v60; // r14
  unsigned __int64 v61; // r13
  __int64 v62; // rdx
  __int64 v63; // rax
  __int64 v64; // rdx
  __int64 v65; // rax
  __int64 v66; // [rsp+8h] [rbp-78h]
  __int64 v67; // [rsp+10h] [rbp-70h]
  __int64 v68; // [rsp+18h] [rbp-68h]
  __int64 v69; // [rsp+20h] [rbp-60h]
  unsigned int v70; // [rsp+20h] [rbp-60h]
  char v71; // [rsp+20h] [rbp-60h]
  __int64 v72; // [rsp+20h] [rbp-60h]
  unsigned int v73; // [rsp+20h] [rbp-60h]
  unsigned int v74; // [rsp+28h] [rbp-58h]
  unsigned int v75; // [rsp+28h] [rbp-58h]
  unsigned __int64 v76; // [rsp+28h] [rbp-58h]
  unsigned int v77; // [rsp+30h] [rbp-50h]
  __int64 v78; // [rsp+30h] [rbp-50h]
  _BYTE *v79; // [rsp+30h] [rbp-50h]
  unsigned __int64 v81; // [rsp+40h] [rbp-40h] BYREF
  __int64 v82; // [rsp+48h] [rbp-38h]

  v5 = *a1;
  if ( (_BYTE)v5 == 14 )
    return 1;
  v6 = a1;
  v7 = a2;
  while ( 1 )
  {
    if ( (unsigned int)(unsigned __int8)v5 - 12 <= 1 )
      return 1;
    switch ( (_BYTE)v5 )
    {
      case 0x11:
        v11 = *((_DWORD *)v6 + 8);
        if ( (v11 & 7) == 0 )
        {
          v12 = v11 >> 3;
          if ( a4 )
          {
            v69 = v12;
            if ( v12 != v7 )
            {
              v74 = v12 - 1;
              for ( i = v7; ; ++i )
              {
                v15 = i;
                if ( *a5 )
                  v15 = v74 - i;
                sub_C440A0(&v81, v6 + 24, 8, (unsigned int)(8 * v15));
                if ( (unsigned int)v82 > 0x40 )
                {
                  v14 = v81;
                  *(_BYTE *)(a3 - v7 + i) = *(_QWORD *)v81;
                  j_j___libc_free_0_0(v14);
                }
                else
                {
                  *(_BYTE *)(a3 - v7 + i) = v81;
                }
                if ( i == v7 + a4 - 1 || v69 == i + 1 )
                  break;
              }
            }
          }
          return 1;
        }
        return 0;
      case 0x12:
        v16 = *(_BYTE *)(*((_QWORD *)v6 + 1) + 8LL);
        if ( v16 == 3 )
        {
          v56 = sub_BD5C60(v6, a2, v5);
          v18 = sub_BCB2E0(v56);
        }
        else if ( v16 == 2 )
        {
          v57 = sub_BD5C60(v6, a2, v5);
          v18 = sub_BCB2D0(v57);
        }
        else
        {
          if ( v16 )
            return 0;
          v17 = sub_BD5C60(v6, a2, v5);
          v18 = sub_BCB2C0(v17);
        }
        a2 = v18;
        v19 = v18;
        v20 = sub_96E500(v6, v18, (__int64)a5);
        if ( v20 )
        {
          v6 = (unsigned __int8 *)v20;
        }
        else
        {
          a2 = v19;
          v6 = (unsigned __int8 *)sub_96F860((__int64)v6, v19, a5, v21, v22, v23);
        }
        goto LABEL_25;
      case 0xA:
        v70 = a4;
        v66 = sub_AE4AC0(a5, *((_QWORD *)v6 + 1));
        v24 = sub_AE1C80(v66, v7);
        v25 = v24;
        v77 = v24;
        v26 = v66 + 16LL * v24 + 24;
        v27 = *(_QWORD *)v26;
        LOBYTE(v26) = *(_BYTE *)(v26 + 8);
        v81 = v27;
        LOBYTE(v82) = v26;
        v28 = (__int64)a5;
        v67 = sub_CA1930(&v81);
        v29 = v7 - v67;
        for ( j = v70; ; j = v35 - v32 )
        {
          v68 = *(_QWORD *)(*(_QWORD *)&v6[32 * (v25 - (*((_DWORD *)v6 + 1) & 0x7FFFFFF))] + 8LL);
          v71 = sub_AE5020(v28, v68);
          v36 = sub_9208B0(v28, v68);
          v82 = v37;
          v81 = ((1LL << v71) + ((unsigned __int64)(v36 + 7) >> 3) - 1) >> v71 << v71;
          if ( v29 < sub_CA1930(&v81)
            && !(unsigned __int8)sub_9704C0(
                                   *(_QWORD *)&v6[32 * (v25 - (*((_DWORD *)v6 + 1) & 0x7FFFFFF))],
                                   v29,
                                   a3,
                                   j,
                                   v28) )
          {
            break;
          }
          if ( ++v77 == *(_DWORD *)(*((_QWORD *)v6 + 1) + 12LL) )
            return 1;
          v25 = v77;
          v31 = *(_BYTE *)(v66 + 16LL * v77 + 32);
          v81 = *(_QWORD *)(v66 + 16LL * v77 + 24);
          LOBYTE(v82) = v31;
          v32 = sub_CA1930(&v81);
          v33 = v67;
          v34 = v32 - (v29 + v67);
          if ( j <= v34 )
            return 1;
          a3 += v34;
          v67 = v32;
          v35 = v33 + v29 + j;
          v29 = 0;
        }
        return 0;
    }
    if ( (v5 & 0xFD) == 9 || (unsigned __int8)(v5 - 15) <= 1u )
      break;
    if ( (_BYTE)v5 != 5 )
      return 0;
    if ( *((_WORD *)v6 + 1) != 48 )
      return 0;
    a2 = *((_QWORD *)v6 + 1);
    v38 = *(_QWORD *)(*(_QWORD *)&v6[-32 * (*((_DWORD *)v6 + 1) & 0x7FFFFFF)] + 8LL);
    if ( v38 != sub_AE4450(a5, a2) )
      return 0;
    v6 = *(unsigned __int8 **)&v6[-32 * (*((_DWORD *)v6 + 1) & 0x7FFFFFF)];
LABEL_25:
    v5 = *v6;
    if ( (_BYTE)v5 == 14 )
      return 1;
  }
  v39 = *((_QWORD *)v6 + 1);
  v75 = a4;
  v40 = *(_QWORD *)(v39 + 24);
  if ( *(_BYTE *)(v39 + 8) == 16 )
  {
    v78 = *(_QWORD *)(v39 + 24);
    v41 = *(_QWORD *)(v39 + 32);
    v42 = sub_AE5020(a5, v40);
    v43 = sub_9208B0((__int64)a5, v78);
    v82 = v44;
    v81 = ((1LL << v42) + ((unsigned __int64)(v43 + 7) >> 3) - 1) >> v42 << v42;
    v45 = sub_CA1930(&v81);
    v46 = v75;
    v47 = v45;
    goto LABEL_44;
  }
  v73 = *(_DWORD *)(v39 + 32);
  v58 = sub_9208B0((__int64)a5, v40);
  v60 = v59;
  v61 = (v58 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  v81 = sub_9208B0((__int64)a5, v40);
  v82 = v62;
  if ( v81 != v61 || (_BYTE)v82 != v60 )
    return 0;
  v41 = v73;
  v63 = sub_9208B0((__int64)a5, v40);
  v82 = v64;
  v81 = (unsigned __int64)(v63 + 7) >> 3;
  v65 = sub_CA1930(&v81);
  v46 = v75;
  v47 = v65;
LABEL_44:
  v48 = v7 / v47;
  v49 = v7 % v47;
  if ( v41 != v7 / v47 )
  {
    v79 = a5;
    v50 = a3;
    v51 = v46;
    v72 = v41;
    v52 = v49;
    v76 = v47;
    v53 = v48;
    while ( 1 )
    {
      v54 = sub_AD69F0(v6, (unsigned int)v53);
      if ( !(unsigned __int8)sub_9704C0(v54, v52, v50, v51, v79) )
        break;
      v55 = v76 - v52;
      if ( v51 > v76 - v52 )
      {
        v51 -= v55;
        v50 += v55;
        ++v53;
        v52 = 0;
        if ( v72 != v53 )
          continue;
      }
      return 1;
    }
    return 0;
  }
  return 1;
}
