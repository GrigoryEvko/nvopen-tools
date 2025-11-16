// Function: sub_1498F00
// Address: 0x1498f00
//
__int64 __fastcall sub_1498F00(_QWORD **a1, __int64 a2, __m128i a3, __m128i a4)
{
  unsigned int v6; // esi
  __int64 *v7; // rdx
  unsigned int v8; // edi
  unsigned int v9; // ecx
  __int64 *v10; // rax
  __int64 v11; // r8
  __int64 result; // rax
  __int64 v13; // r8
  int v14; // edx
  int v15; // esi
  _QWORD *v16; // r8
  unsigned int v17; // ecx
  int v18; // edx
  __int64 *v19; // r12
  __int64 v20; // rdi
  int j; // eax
  int v22; // r15d
  unsigned int v23; // r11d
  __int64 *v24; // rcx
  __int64 v25; // r14
  __int64 v26; // rsi
  __int64 v27; // rsi
  __int64 v28; // r12
  __int64 v29; // rdx
  __int64 v30; // rax
  _QWORD *v31; // r14
  _QWORD *i; // r12
  __int64 v33; // r15
  __int64 v34; // rax
  int v35; // r14d
  __int64 v36; // r12
  __int64 *v37; // r15
  __int64 v38; // rdx
  _QWORD *v39; // r15
  char v40; // r12
  __int64 v41; // r14
  __int64 v42; // r11
  __int64 v43; // rax
  __int64 *v44; // rdi
  _QWORD *v45; // r15
  char v46; // r12
  __int64 v47; // r14
  __int64 v48; // r11
  __int64 v49; // rax
  _QWORD *v50; // r15
  char v51; // r12
  __int64 v52; // r14
  __int64 v53; // r11
  __int64 v54; // rax
  _QWORD *v55; // r15
  char v56; // r12
  __int64 v57; // r14
  __int64 v58; // r11
  __int64 v59; // rax
  __int64 v60; // rsi
  int v61; // edx
  int v62; // edx
  int v63; // ecx
  __int64 *v64; // r8
  _QWORD *v65; // rdi
  int v66; // r11d
  __int64 v67; // r9
  __int64 v68; // rsi
  int v69; // r11d
  unsigned __int64 v70; // r14
  __int64 v71; // rdi
  __int64 *v72; // r15
  __int64 v73; // rdx
  __int64 v74; // rax
  __int64 *v75; // rax
  int v76; // r9d
  __int64 *v77; // r11
  __int64 *v78; // [rsp+8h] [rbp-C8h]
  __int64 v79; // [rsp+10h] [rbp-C0h]
  __int64 v80; // [rsp+10h] [rbp-C0h]
  __int64 v81; // [rsp+10h] [rbp-C0h]
  __int64 v82; // [rsp+10h] [rbp-C0h]
  __int64 v83; // [rsp+18h] [rbp-B8h]
  _QWORD *v84; // [rsp+20h] [rbp-B0h]
  _QWORD *v85; // [rsp+20h] [rbp-B0h]
  _QWORD *v86; // [rsp+20h] [rbp-B0h]
  _QWORD *v87; // [rsp+20h] [rbp-B0h]
  __int64 v88; // [rsp+20h] [rbp-B0h]
  __int64 v89; // [rsp+28h] [rbp-A8h]
  __int64 v90; // [rsp+28h] [rbp-A8h]
  __int64 v91; // [rsp+28h] [rbp-A8h]
  unsigned int v92; // [rsp+28h] [rbp-A8h]
  __int64 v93[2]; // [rsp+30h] [rbp-A0h] BYREF
  _QWORD v94[2]; // [rsp+40h] [rbp-90h] BYREF
  __int64 *v95; // [rsp+50h] [rbp-80h] BYREF
  __int64 v96; // [rsp+58h] [rbp-78h]
  _BYTE v97[112]; // [rsp+60h] [rbp-70h] BYREF

  v6 = *((_DWORD *)a1 + 8);
  v7 = a1[2];
  if ( !v6 )
  {
    switch ( *(_WORD *)(a2 + 24) )
    {
      case 0:
      case 0xB:
        v13 = (__int64)(a1 + 1);
        result = a2;
        goto LABEL_7;
      case 1:
LABEL_23:
        v26 = sub_1498F00(a1, *(_QWORD *)(a2 + 32), v7);
        if ( v26 == *(_QWORD *)(a2 + 32) )
          goto LABEL_17;
        result = sub_14835F0(*a1, v26, *(_QWORD *)(a2 + 40), 0, a3, a4);
        v7 = a1[2];
        v6 = *((_DWORD *)a1 + 8);
        break;
      case 2:
LABEL_25:
        v27 = sub_1498F00(a1, *(_QWORD *)(a2 + 32), v7);
        if ( v27 == *(_QWORD *)(a2 + 32) )
          goto LABEL_17;
        result = sub_14747F0((__int64)*a1, v27, *(_QWORD *)(a2 + 40), 0);
        v7 = a1[2];
        v6 = *((_DWORD *)a1 + 8);
        break;
      case 3:
LABEL_71:
        v60 = sub_1498F00(a1, *(_QWORD *)(a2 + 32), v7);
        if ( v60 == *(_QWORD *)(a2 + 32) )
          goto LABEL_17;
        result = sub_147B0D0((__int64)*a1, v60, *(_QWORD *)(a2 + 40), 0);
        v7 = a1[2];
        v6 = *((_DWORD *)a1 + 8);
        break;
      case 4:
LABEL_57:
        v50 = *(_QWORD **)(a2 + 32);
        v95 = (__int64 *)v97;
        v96 = 0x200000000LL;
        v86 = &v50[*(_QWORD *)(a2 + 40)];
        if ( v50 == v86 )
          goto LABEL_18;
        v51 = 0;
        do
        {
          v52 = *v50;
          v53 = sub_1498F00(a1, *v50, v7);
          v54 = (unsigned int)v96;
          if ( (unsigned int)v96 >= HIDWORD(v96) )
          {
            v81 = v53;
            sub_16CD150(&v95, v97, 0, 8);
            v54 = (unsigned int)v96;
            v53 = v81;
          }
          v7 = v95;
          v95[v54] = v53;
          v44 = v95;
          LODWORD(v96) = v96 + 1;
          ++v50;
          v51 |= v95[(unsigned int)v96 - 1] != v52;
        }
        while ( v86 != v50 );
        result = a2;
        if ( v51 )
        {
          result = (__int64)sub_147DD40((__int64)*a1, (__int64 *)&v95, 0, 0, a3, a4);
          v44 = v95;
        }
        goto LABEL_55;
      case 5:
LABEL_64:
        v55 = *(_QWORD **)(a2 + 32);
        v95 = (__int64 *)v97;
        v96 = 0x200000000LL;
        v87 = &v55[*(_QWORD *)(a2 + 40)];
        if ( v55 == v87 )
          goto LABEL_18;
        v56 = 0;
        do
        {
          v57 = *v55;
          v58 = sub_1498F00(a1, *v55, v7);
          v59 = (unsigned int)v96;
          if ( (unsigned int)v96 >= HIDWORD(v96) )
          {
            v79 = v58;
            sub_16CD150(&v95, v97, 0, 8);
            v59 = (unsigned int)v96;
            v58 = v79;
          }
          v7 = v95;
          v95[v59] = v58;
          v44 = v95;
          LODWORD(v96) = v96 + 1;
          ++v55;
          v56 |= v95[(unsigned int)v96 - 1] != v57;
        }
        while ( v87 != v55 );
        result = a2;
        if ( v56 )
        {
          result = sub_147EE30(*a1, &v95, 0, 0, a3, a4);
          v44 = v95;
        }
        goto LABEL_55;
      case 6:
LABEL_27:
        v28 = sub_1498F00(a1, *(_QWORD *)(a2 + 32), v7);
        v30 = sub_1498F00(a1, *(_QWORD *)(a2 + 40), v29);
        if ( v28 == *(_QWORD *)(a2 + 32) && v30 == *(_QWORD *)(a2 + 40) )
        {
LABEL_17:
          v7 = a1[2];
          v6 = *((_DWORD *)a1 + 8);
LABEL_18:
          result = a2;
        }
        else
        {
          result = sub_1483CF0(*a1, v28, v30, a3, a4);
LABEL_29:
          v7 = a1[2];
          v6 = *((_DWORD *)a1 + 8);
        }
        break;
      case 7:
LABEL_30:
        v31 = *(_QWORD **)(a2 + 32);
        v95 = (__int64 *)v97;
        v96 = 0x800000000LL;
        for ( i = &v31[*(_QWORD *)(a2 + 40)]; i != v31; LODWORD(v96) = v96 + 1 )
        {
          v33 = sub_1498F00(a1, *v31, v7);
          v34 = (unsigned int)v96;
          if ( (unsigned int)v96 >= HIDWORD(v96) )
          {
            sub_16CD150(&v95, v97, 0, 8);
            v34 = (unsigned int)v96;
          }
          v7 = v95;
          ++v31;
          v95[v34] = v33;
        }
        if ( ((unsigned __int8 (__fastcall *)(_QWORD *, __int64, __int64 *))a1[6])(a1[7], a2, v7) )
        {
          if ( *((_DWORD *)a1 + 10) == 1 )
          {
            if ( (int)v96 - 1 > 0 )
            {
              v70 = 8;
              v83 = 8LL * (unsigned int)(v96 - 2) + 16;
              do
              {
                v71 = (__int64)*a1;
                v72 = &v95[v70 / 8 - 1];
                v73 = v95[v70 / 8];
                v74 = *v72;
                v93[0] = (__int64)v94;
                v94[1] = v73;
                v94[0] = v74;
                v93[1] = 0x200000002LL;
                v75 = sub_147DD40(v71, v93, 0, 0, a3, a4);
                if ( (_QWORD *)v93[0] != v94 )
                {
                  v78 = v75;
                  _libc_free(v93[0]);
                  v75 = v78;
                }
                *v72 = (__int64)v75;
                v70 += 8LL;
              }
              while ( v83 != v70 );
            }
          }
          else
          {
            v35 = v96 - 2;
            if ( (int)v96 - 2 >= 0 )
            {
              v36 = v35;
              do
              {
                --v35;
                v37 = &v95[v36];
                v38 = v95[v36-- + 1];
                *v37 = sub_14806B0((__int64)*a1, *v37, v38, 0, 0);
              }
              while ( v35 != -1 );
            }
          }
        }
        result = sub_14785F0((__int64)*a1, &v95, *(_QWORD *)(a2 + 48), 0);
        if ( v95 != (__int64 *)v97 )
        {
          v90 = result;
          _libc_free((unsigned __int64)v95);
          result = v90;
        }
        goto LABEL_29;
      case 8:
LABEL_41:
        v39 = *(_QWORD **)(a2 + 32);
        v95 = (__int64 *)v97;
        v96 = 0x200000000LL;
        v84 = &v39[*(_QWORD *)(a2 + 40)];
        if ( v39 == v84 )
          goto LABEL_18;
        v40 = 0;
        do
        {
          v41 = *v39;
          v42 = sub_1498F00(a1, *v39, v7);
          v43 = (unsigned int)v96;
          if ( (unsigned int)v96 >= HIDWORD(v96) )
          {
            v80 = v42;
            sub_16CD150(&v95, v97, 0, 8);
            v43 = (unsigned int)v96;
            v42 = v80;
          }
          v7 = v95;
          v95[v43] = v42;
          v44 = v95;
          LODWORD(v96) = v96 + 1;
          ++v39;
          v40 |= v95[(unsigned int)v96 - 1] != v41;
        }
        while ( v84 != v39 );
        result = a2;
        if ( v40 )
        {
          result = sub_14813B0(*a1, &v95, a3, a4);
          v44 = v95;
        }
        goto LABEL_55;
      case 9:
LABEL_48:
        v45 = *(_QWORD **)(a2 + 32);
        v95 = (__int64 *)v97;
        v96 = 0x200000000LL;
        v85 = &v45[*(_QWORD *)(a2 + 40)];
        if ( v45 == v85 )
          goto LABEL_18;
        v46 = 0;
        do
        {
          v47 = *v45;
          v48 = sub_1498F00(a1, *v45, v7);
          v49 = (unsigned int)v96;
          if ( (unsigned int)v96 >= HIDWORD(v96) )
          {
            v82 = v48;
            sub_16CD150(&v95, v97, 0, 8);
            v49 = (unsigned int)v96;
            v48 = v82;
          }
          v7 = v95;
          v95[v49] = v48;
          v44 = v95;
          LODWORD(v96) = v96 + 1;
          ++v45;
          v46 |= v95[(unsigned int)v96 - 1] != v47;
        }
        while ( v85 != v45 );
        result = a2;
        if ( v46 )
        {
          result = sub_147A3C0(*a1, &v95, a3, a4);
          v44 = v95;
        }
LABEL_55:
        if ( v44 == (__int64 *)v97 )
          goto LABEL_29;
        v91 = result;
        _libc_free((unsigned __int64)v44);
        v7 = a1[2];
        v6 = *((_DWORD *)a1 + 8);
        result = v91;
        break;
      case 0xA:
        goto LABEL_18;
      default:
        goto LABEL_114;
    }
    v13 = (__int64)(a1 + 1);
    if ( v6 )
    {
      v8 = v6 - 1;
LABEL_21:
      v22 = 1;
      v19 = 0;
      v23 = v8 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v24 = &v7[2 * v23];
      v25 = *v24;
      if ( *v24 == a2 )
        return v24[1];
      while ( v25 != -8 )
      {
        if ( !v19 && v25 == -16 )
          v19 = v24;
        v23 = v8 & (v22 + v23);
        v24 = &v7[2 * v23];
        v25 = *v24;
        if ( *v24 == a2 )
          return v24[1];
        ++v22;
      }
      v61 = *((_DWORD *)a1 + 6);
      if ( !v19 )
        v19 = v24;
      a1[1] = (_QWORD *)((char *)a1[1] + 1);
      v18 = v61 + 1;
      if ( 4 * v18 < 3 * v6 )
      {
        if ( v6 - (v18 + *((_DWORD *)a1 + 7)) > v6 >> 3 )
        {
LABEL_10:
          *((_DWORD *)a1 + 6) = v18;
          if ( *v19 != -8 )
            --*((_DWORD *)a1 + 7);
          *v19 = a2;
          v19[1] = result;
          return result;
        }
        v88 = result;
        v92 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
        sub_1466670(v13, v6);
        v62 = *((_DWORD *)a1 + 8);
        if ( v62 )
        {
          v63 = v62 - 1;
          v64 = 0;
          v65 = a1[2];
          v66 = 1;
          LODWORD(v67) = v63 & v92;
          v18 = *((_DWORD *)a1 + 6) + 1;
          result = v88;
          v19 = &v65[2 * (v63 & v92)];
          v68 = *v19;
          if ( *v19 != a2 )
          {
            while ( v68 != -8 )
            {
              if ( !v64 && v68 == -16 )
                v64 = v19;
              v67 = v63 & (unsigned int)(v66 + v67);
              v19 = &v65[2 * v67];
              v68 = *v19;
              if ( *v19 == a2 )
                goto LABEL_10;
              ++v66;
            }
            if ( v64 )
              v19 = v64;
          }
          goto LABEL_10;
        }
LABEL_114:
        ++*((_DWORD *)a1 + 6);
        BUG();
      }
    }
    else
    {
LABEL_7:
      a1[1] = (_QWORD *)((char *)a1[1] + 1);
      v6 = 0;
    }
    v89 = result;
    sub_1466670(v13, 2 * v6);
    v14 = *((_DWORD *)a1 + 8);
    if ( v14 )
    {
      v15 = v14 - 1;
      v16 = a1[2];
      v17 = (v14 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v18 = *((_DWORD *)a1 + 6) + 1;
      result = v89;
      v19 = &v16[2 * v17];
      v20 = *v19;
      if ( *v19 != a2 )
      {
        v76 = 1;
        v77 = 0;
        while ( v20 != -8 )
        {
          if ( v20 == -16 && !v77 )
            v77 = v19;
          v17 = v15 & (v76 + v17);
          v19 = &v16[2 * v17];
          v20 = *v19;
          if ( *v19 == a2 )
            goto LABEL_10;
          ++v76;
        }
        if ( v77 )
          v19 = v77;
      }
      goto LABEL_10;
    }
    goto LABEL_114;
  }
  v8 = v6 - 1;
  v9 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = &v7[2 * v9];
  v11 = *v10;
  if ( *v10 != a2 )
  {
    for ( j = 1; ; j = v69 )
    {
      if ( v11 == -8 )
      {
        switch ( *(_WORD *)(a2 + 24) )
        {
          case 0:
          case 0xB:
LABEL_73:
            v13 = (__int64)(a1 + 1);
            result = a2;
            goto LABEL_21;
          case 1:
            goto LABEL_23;
          case 2:
            goto LABEL_25;
          case 3:
            goto LABEL_71;
          case 4:
            goto LABEL_57;
          case 5:
            goto LABEL_64;
          case 6:
            goto LABEL_27;
          case 7:
            goto LABEL_30;
          case 8:
            goto LABEL_41;
          case 9:
            goto LABEL_48;
          case 0xA:
            goto LABEL_18;
          default:
            goto LABEL_114;
        }
      }
      v69 = j + 1;
      v9 = v8 & (j + v9);
      v10 = &v7[2 * v9];
      v11 = *v10;
      if ( *v10 == a2 )
        break;
    }
  }
  if ( v10 == &v7[2 * v6] )
  {
    switch ( *(_WORD *)(a2 + 24) )
    {
      case 0:
      case 0xB:
        goto LABEL_73;
      case 1:
        goto LABEL_23;
      case 2:
        goto LABEL_25;
      case 3:
        goto LABEL_71;
      case 4:
        goto LABEL_57;
      case 5:
        goto LABEL_64;
      case 6:
        goto LABEL_27;
      case 7:
        goto LABEL_30;
      case 8:
        goto LABEL_41;
      case 9:
        goto LABEL_48;
      case 0xA:
        goto LABEL_18;
      default:
        goto LABEL_114;
    }
  }
  return v10[1];
}
