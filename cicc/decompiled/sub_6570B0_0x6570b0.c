// Function: sub_6570B0
// Address: 0x6570b0
//
__int64 __fastcall sub_6570B0(__int64 a1)
{
  __int64 v1; // rax
  __int64 result; // rax
  __int64 v3; // rdi
  char v4; // dl
  _QWORD *v5; // r12
  unsigned __int64 v6; // r13
  unsigned __int64 v7; // r14
  __int64 *v8; // rax
  __int64 v9; // r15
  __int64 v10; // rsi
  __int64 i; // rbx
  __int64 *v12; // rax
  unsigned __int64 v13; // rdx
  __int64 *v14; // r11
  unsigned int v15; // edx
  __int64 v16; // rsi
  __int64 j; // rax
  _QWORD *v18; // rcx
  int v19; // r10d
  __int64 v20; // r9
  unsigned int v21; // ecx
  unsigned __int64 v22; // r8
  unsigned int v23; // esi
  unsigned __int64 *v24; // rdx
  unsigned __int64 *v25; // rax
  unsigned __int64 v26; // r14
  __int64 v27; // rsi
  int v28; // r15d
  __int64 v29; // rax
  char v30; // dl
  __int64 v31; // rax
  int v32; // eax
  __int64 v33; // rax
  char v34; // dl
  __int64 v35; // rax
  __int64 v36; // r12
  __int64 v37; // rax
  __int64 v38; // r12
  __int64 v39; // rsi
  __int64 v40; // rax
  __int64 v41; // r12
  char v42; // al
  __int64 v43; // r13
  __int64 v45; // [rsp+10h] [rbp-270h]
  unsigned int v46; // [rsp+18h] [rbp-268h]
  int v47; // [rsp+1Ch] [rbp-264h]
  int v48; // [rsp+20h] [rbp-260h]
  int v49; // [rsp+24h] [rbp-25Ch]
  _QWORD *v50; // [rsp+28h] [rbp-258h]
  unsigned __int64 v51; // [rsp+30h] [rbp-250h]
  unsigned __int64 v52; // [rsp+40h] [rbp-240h]
  unsigned int v53; // [rsp+48h] [rbp-238h]
  int v54; // [rsp+4Ch] [rbp-234h]
  unsigned __int64 v55; // [rsp+58h] [rbp-228h] BYREF
  __int64 *v56; // [rsp+60h] [rbp-220h] BYREF
  __int64 v57; // [rsp+68h] [rbp-218h] BYREF
  _QWORD v58[66]; // [rsp+70h] [rbp-210h] BYREF

  v1 = *(_QWORD *)a1;
  v55 = 0;
  v56 = 0;
  if ( *(_BYTE *)(v1 + 80) != 7 )
    goto LABEL_2;
  v50 = *(_QWORD **)(v1 + 88);
  v3 = v50[15];
  v45 = v3;
  if ( (unsigned int)sub_8D2FB0(v3) )
    v45 = sub_8D46C0(v3);
  v46 = 0;
  v4 = *(_BYTE *)(v45 + 140);
  if ( (v4 & 0xFB) != 8
    || (v3 = v45, v46 = sub_8D4C10(v45, dword_4F077C4 != 2), v4 = *(_BYTE *)(v45 + 140), v29 = v45, v4 != 12) )
  {
    if ( v4 )
      goto LABEL_50;
LABEL_9:
    v5 = (_QWORD *)v50[16];
    if ( !v5 )
      goto LABEL_2;
    v53 = 1;
    v49 = 0;
    v48 = 0;
    v47 = 0;
    goto LABEL_11;
  }
  do
  {
    v29 = *(_QWORD *)(v29 + 160);
    v30 = *(_BYTE *)(v29 + 140);
  }
  while ( v30 == 12 );
  if ( !v30 )
    goto LABEL_9;
LABEL_50:
  v3 = v45;
  v47 = sub_8DBE70(v45);
  if ( v47 )
  {
    v49 = 1;
    v48 = 0;
    v47 = 0;
LABEL_52:
    v5 = (_QWORD *)v50[16];
    if ( v5 )
    {
      v53 = 0;
LABEL_11:
      v51 = 0;
      while ( 1 )
      {
        v6 = v5[2];
        v54 = v53 | v49;
        v57 = 0;
        if ( v53 | v49 )
        {
          v52 = v6 + 64;
          if ( !v53 )
          {
            i = *(_QWORD *)&dword_4D03B80;
            goto LABEL_31;
          }
          goto LABEL_76;
        }
        if ( v55 == v51 )
        {
          v52 = v6 + 64;
          if ( v51 )
          {
            v3 = 2829;
            sub_6851C0(2829, v52);
          }
          else
          {
            v3 = 2828;
            sub_685360(2828, v6 + 64);
          }
LABEL_76:
          v54 = 1;
          v53 = 1;
          i = sub_72C930(v3);
          goto LABEL_31;
        }
        v7 = v51 + 1;
        if ( v48 )
        {
          ++v51;
          v53 = 0;
          i = sub_8D4050(v45);
          v52 = v6 + 64;
          goto LABEL_31;
        }
        if ( v47 )
        {
          v33 = sub_643630((__int64)v50, v45, v51, 0, a1 + 48, (__int64)&v57);
          v34 = *(_BYTE *)(v33 + 140);
          for ( i = v33; v34 == 12; v34 = *(_BYTE *)(v33 + 140) )
            v33 = *(_QWORD *)(v33 + 160);
          ++v51;
          v52 = v6 + 64;
          v53 = v34 == 0;
          v54 = v53;
          goto LABEL_31;
        }
        v8 = v56;
        if ( !v56 )
        {
LABEL_131:
          v56 = 0;
          BUG();
        }
        while ( !*v8 || !v8[1] && (v8[18] & 4) != 0 )
        {
          v8 = (__int64 *)v8[14];
          if ( !v8 )
            goto LABEL_131;
        }
        v9 = v8[15];
        v56 = v8;
        if ( (unsigned int)sub_8D2FB0(v9) )
          v9 = sub_8D46C0(v9);
        v10 = 0;
        if ( !(unsigned int)sub_8D2FB0(v56[15]) )
        {
          v10 = v46 & 0xFFFFFFFE;
          if ( (v56[18] & 0x20) == 0 )
            v10 = v46;
        }
        i = sub_73C570(v9, v10, -1);
        if ( (_DWORD)qword_4F077B4 )
        {
          if ( qword_4F077A0 <= 0x1387Fu )
            goto LABEL_29;
        }
        else if ( dword_4F077BC && qword_4F077A8 <= 0x1387Fu )
        {
LABEL_29:
          v12 = v56;
          v13 = v6 + 64;
          if ( (v56[11] & 3) == 0 )
          {
            v54 = 0;
            v52 = v6 + 64;
            ++v51;
            v53 = 0;
            goto LABEL_31;
          }
          goto LABEL_84;
        }
        v32 = sub_884000(*v56, 1);
        v13 = v6 + 64;
        v52 = v6 + 64;
        if ( !v32 )
        {
          v12 = v56;
LABEL_84:
          v52 = v13;
          sub_6854C0(2845, v13, *v12);
          v51 = v7;
          v54 = 1;
          v53 = 1;
          goto LABEL_31;
        }
        v53 = dword_4F077BC;
        if ( !dword_4F077BC )
          goto LABEL_95;
        v53 = qword_4F077B4;
        if ( !(_DWORD)qword_4F077B4 )
          break;
        v54 = 0;
        ++v51;
        v53 = 0;
LABEL_31:
        *(_BYTE *)(v6 + 175) &= ~1u;
        *(_QWORD *)(v6 + 120) = i;
        v14 = qword_4D03BF0;
        v15 = *((_DWORD *)qword_4D03BF0 + 2);
        v16 = *qword_4D03BF0;
        for ( j = v15 & (unsigned int)(v6 >> 3); ; j = v15 & ((_DWORD)j + 1) )
        {
          v18 = (_QWORD *)(v16 + 16LL * (unsigned int)j);
          if ( v6 == *v18 )
            break;
        }
        *v18 = 0;
        if ( *(_QWORD *)(v16 + 16LL * (((_DWORD)j + 1) & v15)) )
        {
          v19 = *((_DWORD *)v14 + 2);
          v20 = *v14;
          v21 = v19 & (j + 1);
          v22 = *(_QWORD *)(*v14 + 16LL * v21);
          while ( 1 )
          {
            v23 = v19 & (v22 >> 3);
            v24 = (unsigned __int64 *)(v20 + 16LL * (v19 & (v21 + 1)));
            if ( v23 <= (unsigned int)j && (v21 < v23 || v21 > (unsigned int)j) || v21 > (unsigned int)j && v21 < v23 )
            {
              v25 = (unsigned __int64 *)(v20 + 16 * j);
              v26 = *v25;
              v27 = v20 + 16LL * v21;
              if ( *v25 )
              {
                *v25 = v22;
                v28 = *((_DWORD *)v25 + 2);
                if ( v22 )
                  *((_DWORD *)v25 + 2) = *(_DWORD *)(v27 + 8);
                *(_QWORD *)v27 = v26;
                *(_DWORD *)(v27 + 8) = v28;
              }
              else
              {
                *v25 = v22;
                if ( v22 )
                  *((_DWORD *)v25 + 2) = *(_DWORD *)(v27 + 8);
                *(_QWORD *)v27 = 0;
              }
              v22 = *v24;
              if ( !*v24 )
                break;
              j = v21;
            }
            else
            {
              v22 = *v24;
              if ( !*v24 )
                break;
            }
            v21 = v19 & (v21 + 1);
          }
        }
        --*((_DWORD *)v14 + 3);
        memset(v58, 0, 0x1D8u);
        v58[19] = v58;
        v58[3] = *(_QWORD *)&dword_4F063F8;
        if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
          BYTE2(v58[22]) |= 1u;
        v31 = *(_QWORD *)v6;
        v58[36] = i;
        v58[34] = i;
        v58[0] = v31;
        v58[6] = *(_QWORD *)(v6 + 64);
        BYTE5(v58[33]) = *(_BYTE *)(v6 + 136);
        *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 624) = v58;
        if ( !v54 )
        {
          if ( v48 )
          {
            sub_693280(v50, v6, v51 - 1);
          }
          else if ( v47 )
          {
            sub_63AEF0(v6, v57);
            sub_8756B0(*(_QWORD *)v6);
          }
          else
          {
            sub_6933C0(v50, v6, v56);
            v56 = (__int64 *)v56[14];
          }
        }
        if ( v57 )
          sub_6E1990(v57);
        v3 = 12;
        sub_8767A0(12, *v50, v52, 1);
        *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 624) = 0;
        v5 = (_QWORD *)*v5;
        if ( !v5 )
          goto LABEL_65;
      }
      if ( qword_4F077A8 )
      {
        ++v51;
        v54 = 0;
        if ( (v56[11] & 3) != 1 || (unsigned int)sub_87D890(*(_QWORD *)(v56[5] + 32)) )
          goto LABEL_31;
        v12 = v56;
        v13 = v6 + 64;
        goto LABEL_84;
      }
LABEL_95:
      v54 = 0;
      ++v51;
      goto LABEL_31;
    }
    v51 = 0;
LABEL_66:
    if ( v55 > v51 )
      sub_6851C0(2830, a1 + 48);
    v50[1] = 0;
    goto LABEL_2;
  }
  v3 = v45;
  v48 = sub_8D3410(v45);
  if ( v48 )
  {
    v35 = v45;
    if ( *(_BYTE *)(v45 + 140) == 12 )
    {
      do
        v35 = *(_QWORD *)(v35 + 160);
      while ( *(_BYTE *)(v35 + 140) == 12 );
    }
    else
    {
      v35 = v45;
    }
    v49 = 0;
    v48 = 1;
    v55 = *(_QWORD *)(v35 + 176);
    goto LABEL_52;
  }
  v36 = sub_8866D0("tuple_size");
  if ( v36 )
  {
    v58[0] = sub_725090(0);
    *(_QWORD *)(v58[0] + 32LL) = v45;
    v37 = sub_8AF060(v36, v58);
    if ( v37 )
    {
      if ( *(_BYTE *)(v37 + 80) == 4 )
      {
        v38 = *(_QWORD *)(v37 + 88);
        if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(v38) )
          sub_8AE000(v38);
        if ( (*(_BYTE *)(v38 + 141) & 0x20) == 0 )
        {
          v39 = v38;
          v40 = sub_879C70("value");
          v41 = v40;
          if ( v40 )
          {
            v42 = *(_BYTE *)(v40 + 80);
            if ( v42 == 2 )
            {
              v43 = *(_QWORD *)(v41 + 88);
            }
            else
            {
              if ( v42 != 7 && v42 != 9 )
                goto LABEL_110;
              v43 = sub_6EA7C0(*(_QWORD *)(v41 + 88), v39);
            }
            if ( v43 )
            {
              if ( (unsigned int)sub_8D2960(*(_QWORD *)(v43 + 128)) )
              {
                v55 = sub_620FD0(v43, &v57);
                v49 = v57;
                if ( !(_DWORD)v57 )
                {
                  v3 = v41;
                  v48 = sub_884000(v41, 1);
                  if ( v48 )
                  {
                    v48 = 0;
                  }
                  else
                  {
                    v3 = 265;
                    sub_6854C0(265, dword_4F07508, v41);
                  }
                  v47 = 1;
                  goto LABEL_52;
                }
              }
            }
          }
LABEL_110:
          sub_685360(2839, dword_4F07508);
          v53 = 1;
          goto LABEL_111;
        }
      }
    }
  }
  v53 = 0;
LABEL_111:
  v3 = v45;
  v47 = sub_643950(v45, (__int64 *)&v55, &v56, 0, a1 + 48);
  v5 = (_QWORD *)v50[16];
  if ( !v47 )
  {
    if ( !v5 )
      goto LABEL_2;
    v49 = 0;
    v48 = 0;
    v53 = 1;
    goto LABEL_11;
  }
  if ( v5 )
  {
    v49 = 0;
    v47 = 0;
    goto LABEL_11;
  }
  v51 = 0;
LABEL_65:
  if ( !v53 )
    goto LABEL_66;
LABEL_2:
  if ( word_4F06418[0] == 67 || (result = a1, *(char *)(a1 + 121) < 0) )
  {
    sub_6851C0(2954, a1 + 48);
    result = sub_72C930(2954);
    *(_WORD *)(a1 + 124) &= 0xF47Fu;
    *(_QWORD *)(a1 + 272) = result;
  }
  return result;
}
