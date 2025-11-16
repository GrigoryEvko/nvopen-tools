// Function: sub_2988EA0
// Address: 0x2988ea0
//
void __fastcall sub_2988EA0(__int64 a1)
{
  __int64 v2; // r9
  __int64 v3; // rdi
  _QWORD *v4; // rdx
  int v5; // esi
  __int64 v6; // r8
  _QWORD *i; // rax
  _QWORD *v8; // rax
  unsigned __int64 v9; // rsi
  _BYTE *v10; // r13
  _QWORD *v11; // r12
  _BYTE *v12; // rbx
  __int64 (__fastcall *v13)(__int64); // rax
  _QWORD *v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rdi
  _QWORD *v17; // rbx
  __int64 (__fastcall *v18)(_QWORD *); // rax
  _QWORD *v19; // rdi
  _BYTE *v20; // r14
  __int64 v21; // rdi
  _BYTE *v22; // rdi
  _QWORD **v23; // r12
  _QWORD **v24; // rbx
  _QWORD *v25; // rdi
  _QWORD *v26; // rax
  __int64 v27; // r11
  _QWORD *v28; // r14
  _QWORD *v29; // rbx
  __int64 v30; // r15
  __int64 v31; // r8
  unsigned int v32; // r13d
  bool v33; // al
  __int64 v34; // r13
  unsigned __int8 v35; // al
  unsigned int v36; // edx
  __int64 v37; // rax
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // rax
  unsigned __int64 v41; // rdx
  int v42; // eax
  bool v43; // r13
  __int64 v44; // rax
  unsigned int v45; // r13d
  __int64 v46; // rdx
  __int64 v47; // rdx
  _BYTE *v48; // rax
  unsigned int v49; // r13d
  bool v50; // al
  __int64 v51; // r13
  __int64 v52; // rdx
  _BYTE *v53; // rax
  unsigned int v54; // r13d
  unsigned int v55; // ebx
  char v56; // r12
  unsigned __int8 *v57; // rax
  unsigned int v58; // r12d
  char v59; // cl
  int v60; // [rsp+Ch] [rbp-164h]
  int v61; // [rsp+Ch] [rbp-164h]
  _QWORD *v62; // [rsp+10h] [rbp-160h]
  _QWORD *v63; // [rsp+18h] [rbp-158h]
  int v64; // [rsp+20h] [rbp-150h]
  __int64 v65; // [rsp+20h] [rbp-150h]
  unsigned __int8 *v66; // [rsp+20h] [rbp-150h]
  _QWORD *v67; // [rsp+20h] [rbp-150h]
  _QWORD *v68; // [rsp+28h] [rbp-148h]
  _BYTE v69[16]; // [rsp+40h] [rbp-130h] BYREF
  __int64 (__fastcall *v70)(__int64); // [rsp+50h] [rbp-120h]
  __int64 v71; // [rsp+58h] [rbp-118h]
  _BYTE v72[16]; // [rsp+60h] [rbp-110h] BYREF
  __int64 (__fastcall *v73)(_QWORD *); // [rsp+70h] [rbp-100h]
  __int64 v74; // [rsp+78h] [rbp-F8h]
  _BYTE *v75; // [rsp+80h] [rbp-F0h] BYREF
  __int64 v76; // [rsp+88h] [rbp-E8h]
  _BYTE v77[48]; // [rsp+90h] [rbp-E0h] BYREF
  _QWORD v78[2]; // [rsp+C0h] [rbp-B0h] BYREF
  _QWORD *v79; // [rsp+D0h] [rbp-A0h]
  _QWORD *v80; // [rsp+D8h] [rbp-98h]
  __int64 v81; // [rsp+E0h] [rbp-90h]
  unsigned __int64 v82; // [rsp+E8h] [rbp-88h]
  _QWORD *v83; // [rsp+F0h] [rbp-80h]
  _QWORD *v84; // [rsp+F8h] [rbp-78h]
  __int64 v85; // [rsp+100h] [rbp-70h]
  __int64 v86; // [rsp+108h] [rbp-68h]
  _QWORD *v87; // [rsp+110h] [rbp-60h]
  _QWORD *v88; // [rsp+118h] [rbp-58h]
  __int64 v89; // [rsp+120h] [rbp-50h]
  unsigned __int64 v90; // [rsp+128h] [rbp-48h]
  _QWORD *v91; // [rsp+130h] [rbp-40h]
  _QWORD *v92; // [rsp+138h] [rbp-38h]

  v2 = a1 + 768;
  v3 = a1 + 624;
  v4 = *(_QWORD **)(a1 + 776);
  v75 = v77;
  v5 = *(_DWORD *)(a1 + 784);
  v76 = 0x600000000LL;
  v6 = *(_QWORD *)(a1 + 768);
  v63 = &v4[5 * *(unsigned int *)(a1 + 792)];
  if ( v5 )
  {
    for ( i = &v4[5 * *(unsigned int *)(a1 + 792)]; v4 != i; v4 += 5 )
    {
      if ( *v4 != -4096 && *v4 != -8192 )
        break;
    }
  }
  else
  {
    v4 += 5 * *(unsigned int *)(a1 + 792);
  }
  v8 = *(_QWORD **)(a1 + 632);
  v68 = &v8[5 * *(unsigned int *)(a1 + 648)];
  v9 = *(_QWORD *)(a1 + 624);
  if ( *(_DWORD *)(a1 + 640) )
  {
    for ( ; v68 != v8; v8 += 5 )
    {
      if ( *v8 != -8192 && *v8 != -4096 )
        break;
    }
  }
  else
  {
    v8 += 5 * *(unsigned int *)(a1 + 648);
  }
  v79 = v4;
  v78[0] = v2;
  v78[1] = v6;
  v80 = v63;
  v81 = v3;
  v82 = v9;
  v83 = v8;
  v84 = v68;
  v85 = v2;
  v86 = v6;
  v87 = v63;
  v88 = v63;
  v89 = v3;
  v90 = v9;
  v91 = v68;
  v92 = v68;
  if ( v68 == v8 )
    goto LABEL_20;
  do
  {
    do
    {
      v10 = v69;
      v71 = 0;
      v70 = sub_29882A0;
      v11 = v78;
      v12 = v69;
      v13 = sub_2988280;
      v14 = v78;
      if ( ((unsigned __int8)sub_2988280 & 1) == 0 )
        goto LABEL_8;
      while ( 1 )
      {
        v13 = *(__int64 (__fastcall **)(__int64))((char *)v13 + *v14 - 1);
LABEL_8:
        v15 = v13((__int64)v14);
        if ( v15 )
          break;
        while ( 1 )
        {
          v10 += 16;
          if ( v72 == v10 )
            goto LABEL_124;
          v16 = *((_QWORD *)v12 + 3);
          v13 = (__int64 (__fastcall *)(__int64))*((_QWORD *)v12 + 2);
          v12 = v10;
          v14 = (_QWORD *)((char *)v78 + v16);
          if ( ((unsigned __int8)v13 & 1) != 0 )
            break;
          v15 = ((__int64 (__fastcall *)(_QWORD *, unsigned __int64, __int64))v13)(v14, v9, v15);
          if ( v15 )
            goto LABEL_12;
        }
      }
LABEL_12:
      if ( *(_DWORD *)(v15 + 24) )
      {
        v26 = *(_QWORD **)(v15 + 16);
        v27 = 4LL * *(unsigned int *)(v15 + 32);
        v28 = &v26[v27];
        if ( v26 != &v26[v27] )
        {
          while ( 1 )
          {
            v29 = v26;
            if ( *v26 != -4096 && *v26 != -8192 )
              break;
            v26 += 4;
            if ( v28 == v26 )
              goto LABEL_13;
          }
          if ( v28 != v26 )
          {
            v30 = v26[1];
            if ( *(_BYTE *)v30 == 59 )
              goto LABEL_42;
LABEL_35:
            while ( 1 )
            {
              v29 += 4;
              if ( v29 == v28 )
                break;
              while ( *v29 == -8192 || *v29 == -4096 )
              {
                v29 += 4;
                if ( v28 == v29 )
                  goto LABEL_13;
              }
              if ( v28 == v29 )
                break;
              v30 = v29[1];
              if ( *(_BYTE *)v30 == 59 )
              {
LABEL_42:
                v31 = *(_QWORD *)(v30 - 64);
                if ( *(_BYTE *)v31 == 17 )
                {
                  v32 = *(_DWORD *)(v31 + 32);
                  if ( !v32 )
                    goto LABEL_71;
                  if ( v32 <= 0x40 )
                    v33 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v32) == *(_QWORD *)(v31 + 24);
                  else
                    v33 = v32 == (unsigned int)sub_C445E0(v31 + 24);
                  goto LABEL_46;
                }
                v51 = *(_QWORD *)(v31 + 8);
                v52 = (unsigned int)*(unsigned __int8 *)(v51 + 8) - 17;
                if ( (unsigned int)v52 > 1 || *(_BYTE *)v31 > 0x15u )
                  goto LABEL_47;
                v9 = 0;
                v66 = *(unsigned __int8 **)(v30 - 64);
                v53 = sub_AD7630((__int64)v66, 0, v52);
                if ( !v53 || *v53 != 17 )
                {
                  if ( *(_BYTE *)(v51 + 8) == 17 )
                  {
                    v42 = *(_DWORD *)(v51 + 32);
                    v9 = 0;
                    v43 = 0;
                    v60 = v42;
                    if ( v42 )
                    {
                      while ( 1 )
                      {
                        v44 = sub_AD69F0(v66, v9);
                        v9 = (unsigned int)v9;
                        if ( !v44 )
                          break;
                        if ( *(_BYTE *)v44 != 13 )
                        {
                          if ( *(_BYTE *)v44 != 17 )
                            break;
                          v45 = *(_DWORD *)(v44 + 32);
                          if ( v45 )
                          {
                            if ( v45 <= 0x40 )
                            {
                              v43 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v45) == *(_QWORD *)(v44 + 24);
                            }
                            else
                            {
                              v9 = (unsigned int)v9;
                              v43 = v45 == (unsigned int)sub_C445E0(v44 + 24);
                            }
                            if ( !v43 )
                              break;
                          }
                          else
                          {
                            v43 = 1;
                          }
                        }
                        v9 = (unsigned int)(v9 + 1);
                        if ( v60 == (_DWORD)v9 )
                        {
                          if ( !v43 )
                            break;
                          goto LABEL_71;
                        }
                      }
                    }
                  }
LABEL_47:
                  v34 = *(_QWORD *)(v30 - 32);
                  v35 = *(_BYTE *)v34;
                  goto LABEL_48;
                }
                v54 = *((_DWORD *)v53 + 8);
                if ( v54 )
                {
                  if ( v54 <= 0x40 )
                    v33 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v54) == *((_QWORD *)v53 + 3);
                  else
                    v33 = v54 == (unsigned int)sub_C445E0((__int64)(v53 + 24));
LABEL_46:
                  if ( !v33 )
                    goto LABEL_47;
                }
LABEL_71:
                v34 = *(_QWORD *)(v30 - 32);
                v46 = *(_QWORD *)(v34 + 16);
                v35 = *(_BYTE *)v34;
                if ( !v46 || *(_QWORD *)(v46 + 8) )
                {
LABEL_48:
                  if ( v35 != 17 )
                    goto LABEL_75;
LABEL_49:
                  v36 = *(_DWORD *)(v34 + 32);
                  if ( v36 )
                  {
                    if ( v36 <= 0x40 )
                    {
                      v50 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v36) == *(_QWORD *)(v34 + 24);
LABEL_82:
                      if ( v50 )
                      {
                        v34 = *(_QWORD *)(v30 - 64);
                        v37 = *(_QWORD *)(v34 + 16);
                        if ( v37 )
                          goto LABEL_53;
                      }
                    }
                    else
                    {
                      v64 = *(_DWORD *)(v34 + 32);
                      if ( v64 == (unsigned int)sub_C445E0(v34 + 24) )
                        goto LABEL_52;
                    }
                  }
                  else
                  {
LABEL_52:
                    v34 = *(_QWORD *)(v30 - 64);
                    v37 = *(_QWORD *)(v34 + 16);
                    if ( v37 )
                    {
LABEL_53:
                      if ( !*(_QWORD *)(v37 + 8) && *(_BYTE *)v34 > 0x1Cu )
                        goto LABEL_55;
                    }
                  }
                }
                else if ( v35 > 0x1Cu )
                {
LABEL_55:
                  if ( *(_QWORD *)(v30 + 16) && (unsigned __int8)(*(_BYTE *)v34 - 82) <= 1u )
                  {
                    v9 = v34;
                    *(_WORD *)(v34 + 2) = sub_B52870(*(_WORD *)(v34 + 2) & 0x3F) | *(_WORD *)(v34 + 2) & 0xFFC0;
                    sub_BD84D0(v30, v34);
                    v40 = (unsigned int)v76;
                    v41 = (unsigned int)v76 + 1LL;
                    if ( v41 > HIDWORD(v76) )
                    {
                      v9 = (unsigned __int64)v77;
                      sub_C8D5F0((__int64)&v75, v77, v41, 8u, v38, v39);
                      v40 = (unsigned int)v76;
                    }
                    *(_QWORD *)&v75[8 * v40] = v30;
                    LODWORD(v76) = v76 + 1;
                  }
                }
                else
                {
                  if ( v35 == 17 )
                    goto LABEL_49;
LABEL_75:
                  v65 = *(_QWORD *)(v34 + 8);
                  v47 = (unsigned int)*(unsigned __int8 *)(v65 + 8) - 17;
                  if ( (unsigned int)v47 <= 1 && v35 <= 0x15u )
                  {
                    v9 = 0;
                    v48 = sub_AD7630(v34, 0, v47);
                    if ( v48 && *v48 == 17 )
                    {
                      v49 = *((_DWORD *)v48 + 8);
                      if ( !v49 )
                        goto LABEL_52;
                      if ( v49 > 0x40 )
                      {
                        v50 = v49 == (unsigned int)sub_C445E0((__int64)(v48 + 24));
                        goto LABEL_82;
                      }
                      if ( *((_QWORD *)v48 + 3) == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v49) )
                        goto LABEL_52;
                    }
                    else if ( *(_BYTE *)(v65 + 8) == 17 )
                    {
                      v61 = *(_DWORD *)(v65 + 32);
                      if ( v61 )
                      {
                        v67 = v29;
                        v62 = v11;
                        v55 = 0;
                        v56 = 0;
                        do
                        {
                          v9 = v55;
                          v57 = (unsigned __int8 *)sub_AD69F0((unsigned __int8 *)v34, v55);
                          if ( !v57 )
                          {
LABEL_108:
                            v29 = v67;
                            v11 = v62;
                            goto LABEL_35;
                          }
                          v9 = *v57;
                          if ( (_BYTE)v9 != 13 )
                          {
                            if ( (_BYTE)v9 != 17 )
                              goto LABEL_108;
                            v58 = *((_DWORD *)v57 + 8);
                            if ( v58 )
                            {
                              if ( v58 <= 0x40 )
                              {
                                v9 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v58);
                                if ( *((_QWORD *)v57 + 3) != v9 )
                                  goto LABEL_108;
                              }
                              else if ( v58 != (unsigned int)sub_C445E0((__int64)(v57 + 24)) )
                              {
                                goto LABEL_108;
                              }
                            }
                            v56 = 1;
                          }
                          ++v55;
                        }
                        while ( v61 != v55 );
                        v59 = v56;
                        v29 = v67;
                        v11 = v62;
                        if ( v59 )
                          goto LABEL_52;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
LABEL_13:
      v17 = v72;
      v74 = 0;
      v73 = sub_2988BD0;
      v18 = sub_2988B80;
      v19 = v11;
      v20 = v72;
      if ( ((unsigned __int8)sub_2988B80 & 1) != 0 )
LABEL_14:
        v18 = *(__int64 (__fastcall **)(_QWORD *))((char *)v18 + *v19 - 1);
      if ( !(unsigned __int8)v18(v19) )
      {
        while ( 1 )
        {
          v17 += 2;
          if ( &v75 == v17 )
            break;
          v21 = *((_QWORD *)v20 + 3);
          v18 = (__int64 (__fastcall *)(_QWORD *))*((_QWORD *)v20 + 2);
          v20 = v17;
          v19 = (_QWORD *)((char *)v11 + v21);
          if ( ((unsigned __int8)v18 & 1) != 0 )
            goto LABEL_14;
          if ( (unsigned __int8)v18(v19) )
            goto LABEL_19;
        }
LABEL_124:
        BUG();
      }
LABEL_19:
      ;
    }
    while ( v68 != v83 );
LABEL_20:
    ;
  }
  while ( v79 != v63 || v68 != v91 || v87 != v63 );
  v22 = v75;
  v23 = (_QWORD **)&v75[8 * (unsigned int)v76];
  v24 = (_QWORD **)v75;
  if ( v23 != (_QWORD **)v75 )
  {
    do
    {
      v25 = *v24++;
      sub_B43D60(v25);
    }
    while ( v23 != v24 );
    v22 = v75;
  }
  if ( v22 != v77 )
    _libc_free((unsigned __int64)v22);
}
