// Function: sub_297A990
// Address: 0x297a990
//
__int64 __fastcall sub_297A990(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, _BYTE *a6)
{
  __int64 v6; // rbx
  unsigned int v7; // r14d
  __int64 v8; // rdx
  __int64 v9; // r12
  int v10; // eax
  __int64 v11; // rax
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  __int64 **v16; // r10
  __int64 v17; // r13
  unsigned __int8 **v18; // rsi
  int v19; // ecx
  unsigned __int8 **v20; // r11
  unsigned __int64 v21; // rax
  __int64 v22; // rax
  int v23; // edx
  int v24; // r13d
  char v25; // al
  char v26; // di
  char **v27; // rax
  __int64 v28; // r14
  __int64 v29; // r12
  __int64 v30; // r13
  char *v31; // r13
  char **v32; // rax
  char **v33; // rdx
  unsigned int v34; // r12d
  __int64 v36; // rax
  __int64 v37; // rax
  char **v38; // rdx
  char **v39; // r13
  char **i; // r12
  char *v41; // rsi
  char **v42; // rax
  char **v43; // rdx
  bool v44; // of
  unsigned __int64 v45; // rax
  unsigned __int64 v46; // rsi
  __int64 v47; // [rsp-8h] [rbp-128h]
  __int64 v48; // [rsp+0h] [rbp-120h]
  __int64 v51; // [rsp+18h] [rbp-108h]
  __int64 v52; // [rsp+20h] [rbp-100h]
  __int64 **v53; // [rsp+28h] [rbp-F8h]
  __int64 v54; // [rsp+30h] [rbp-F0h]
  __int64 v55; // [rsp+38h] [rbp-E8h]
  __int64 v56; // [rsp+38h] [rbp-E8h]
  __int64 v58; // [rsp+58h] [rbp-C8h]
  unsigned __int8 **v59; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v60; // [rsp+68h] [rbp-B8h]
  _BYTE v61[32]; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v62; // [rsp+90h] [rbp-90h] BYREF
  char **v63; // [rsp+98h] [rbp-88h]
  __int64 v64; // [rsp+A0h] [rbp-80h]
  int v65; // [rsp+A8h] [rbp-78h]
  char v66; // [rsp+ACh] [rbp-74h]
  char v67; // [rsp+B0h] [rbp-70h] BYREF

  v6 = *(_QWORD *)(a2 + 56);
  v63 = (char **)&v67;
  v62 = 0;
  v64 = 8;
  v65 = 0;
  v66 = 1;
  v58 = a2 + 48;
  if ( v6 == a2 + 48 )
    return 1;
  v7 = 0;
  v54 = 0;
LABEL_3:
  while ( 2 )
  {
    if ( !v6 )
      BUG();
    v8 = *(unsigned __int8 *)(v6 - 24);
    v9 = v6 - 24;
    if ( (unsigned __int8)v8 <= 0x1Cu )
    {
      if ( (_BYTE)v8 != 5 )
      {
        v26 = v66;
        goto LABEL_22;
      }
      v10 = *(unsigned __int16 *)(v6 - 22);
    }
    else
    {
      v10 = (unsigned __int8)v8 - 29;
    }
    switch ( v10 )
    {
      case 12:
      case 13:
      case 14:
      case 15:
      case 16:
      case 17:
      case 18:
      case 21:
      case 24:
      case 25:
      case 26:
      case 27:
      case 28:
      case 29:
      case 30:
      case 34:
      case 38:
      case 39:
      case 40:
      case 41:
      case 42:
      case 43:
      case 44:
      case 45:
      case 46:
      case 47:
      case 48:
      case 49:
      case 50:
      case 53:
      case 54:
      case 56:
      case 57:
      case 61:
      case 62:
      case 63:
      case 64:
      case 65:
      case 67:
        v11 = 32LL * (*(_DWORD *)(v6 - 20) & 0x7FFFFFF);
        if ( (*(_BYTE *)(v6 - 17) & 0x40) != 0 )
        {
          v12 = *(_QWORD *)(v6 - 32);
          v13 = v12 + v11;
        }
        else
        {
          v12 = v9 - v11;
          v13 = v6 - 24;
        }
        v14 = v13 - v12;
        v59 = (unsigned __int8 **)v61;
        v15 = v14 >> 5;
        v16 = *(__int64 ***)(a1 + 8);
        v60 = 0x400000000LL;
        v17 = v14 >> 5;
        if ( (unsigned __int64)v14 > 0x80 )
        {
          v51 = v14;
          v52 = v12;
          v53 = v16;
          v56 = v14 >> 5;
          sub_C8D5F0((__int64)&v59, v61, v15, 8u, v12, (__int64)v61);
          v20 = v59;
          v19 = v60;
          LODWORD(v15) = v56;
          v16 = v53;
          v12 = v52;
          v18 = &v59[(unsigned int)v60];
          v14 = v51;
        }
        else
        {
          v18 = (unsigned __int8 **)v61;
          v19 = 0;
          v20 = (unsigned __int8 **)v61;
        }
        if ( v14 > 0 )
        {
          v21 = 0;
          do
          {
            v18[v21 / 8] = *(unsigned __int8 **)(v12 + 4 * v21);
            v21 += 8LL;
            --v17;
          }
          while ( v17 );
          v20 = v59;
          v19 = v60;
        }
        LODWORD(v60) = v19 + v15;
        v22 = sub_DFCEF0(v16, (unsigned __int8 *)(v6 - 24), v20, (unsigned int)(v19 + v15), 3);
        a6 = v61;
        v55 = v22;
        v24 = v23;
        if ( v59 != (unsigned __int8 **)v61 )
          _libc_free((unsigned __int64)v59);
        if ( v24 || (v25 = sub_991A70((unsigned __int8 *)(v6 - 24), 0, 0, 0, 0, 1u, 0), a4 = v47, !v25) )
        {
          v8 = *(unsigned __int8 *)(v6 - 24);
LABEL_20:
          v26 = v66;
          goto LABEL_21;
        }
        if ( *(_BYTE *)(v6 - 24) == 85 )
        {
          v36 = *(_QWORD *)(v6 - 56);
          if ( v36 )
          {
            if ( !*(_BYTE *)v36 && *(_QWORD *)(v36 + 24) == *(_QWORD *)(v6 + 56) && (*(_BYTE *)(v36 + 33) & 0x20) != 0 )
            {
              v8 = (unsigned int)(*(_DWORD *)(v36 + 36) - 68);
              if ( (unsigned int)v8 <= 3 )
              {
                v26 = v66;
                goto LABEL_51;
              }
            }
          }
        }
        v37 = 4LL * (*(_DWORD *)(v6 - 20) & 0x7FFFFFF);
        if ( (*(_BYTE *)(v6 - 17) & 0x40) != 0 )
        {
          v38 = *(char ***)(v6 - 32);
          a4 = (__int64)&v38[v37];
        }
        else
        {
          a4 = v6 - 24;
          v38 = (char **)(v9 - v37 * 8);
        }
        if ( (char **)a4 != v38 )
        {
          v39 = (char **)a4;
          for ( i = v38; v39 != i; i += 4 )
          {
            v41 = *i;
            if ( *i && (unsigned __int8)*v41 > 0x1Cu )
            {
              v26 = v66;
              if ( v66 )
              {
                v42 = v63;
                v43 = &v63[HIDWORD(v64)];
                if ( v63 != v43 )
                {
                  while ( v41 != *v42 )
                  {
                    if ( v43 == ++v42 )
                      goto LABEL_74;
                  }
                  v9 = v6 - 24;
LABEL_70:
                  v8 = *(unsigned __int8 *)(v6 - 24);
LABEL_21:
                  if ( (_BYTE)v8 == 85 && (v36 = *(_QWORD *)(v6 - 56)) != 0 )
                  {
LABEL_51:
                    if ( *(_BYTE *)v36
                      || *(_QWORD *)(v36 + 24) != *(_QWORD *)(v6 + 56)
                      || (*(_BYTE *)(v36 + 33) & 0x20) == 0
                      || (unsigned int)(*(_DWORD *)(v36 + 36) - 68) > 3 )
                    {
                      goto LABEL_22;
                    }
                  }
                  else
                  {
LABEL_22:
                    ++v7;
                  }
                  if ( v7 > (unsigned int)qword_5007208 )
                    goto LABEL_78;
                  if ( !v26 )
                    goto LABEL_42;
                  v27 = v63;
                  a4 = HIDWORD(v64);
                  v8 = (__int64)&v63[HIDWORD(v64)];
                  if ( v63 == (char **)v8 )
                  {
LABEL_44:
                    if ( HIDWORD(v64) < (unsigned int)v64 )
                    {
                      a4 = (unsigned int)++HIDWORD(v64);
                      *(_QWORD *)v8 = v9;
                      v6 = *(_QWORD *)(v6 + 8);
                      ++v62;
                      if ( v58 == v6 )
                        goto LABEL_30;
                      goto LABEL_3;
                    }
LABEL_42:
                    sub_C8CC70((__int64)&v62, v9, v8, a4, a5, (__int64)a6);
                    v6 = *(_QWORD *)(v6 + 8);
                    if ( v58 == v6 )
                      goto LABEL_30;
                    goto LABEL_3;
                  }
                  while ( (char *)v9 != *v27 )
                  {
                    if ( (char **)v8 == ++v27 )
                      goto LABEL_44;
                  }
                  v6 = *(_QWORD *)(v6 + 8);
                  if ( v58 != v6 )
                    goto LABEL_3;
LABEL_30:
                  v26 = v66;
                  v28 = *(_QWORD *)(a2 + 56);
                  if ( v58 == v28 )
                  {
LABEL_38:
                    v34 = 1;
                    goto LABEL_39;
                  }
                  v29 = v48;
                  while ( 2 )
                  {
                    while ( 1 )
                    {
                      v30 = v28;
                      v28 = *(_QWORD *)(v28 + 8);
                      v31 = (char *)(v30 - 24);
                      if ( !v26 )
                        break;
                      v32 = v63;
                      v33 = &v63[HIDWORD(v64)];
                      if ( v63 == v33 )
                        goto LABEL_80;
                      while ( v31 != *v32 )
                      {
                        if ( v33 == ++v32 )
                          goto LABEL_80;
                      }
                      if ( v6 == v28 )
                        goto LABEL_38;
                    }
                    if ( !sub_C8CA60((__int64)&v62, (__int64)v31) )
                    {
LABEL_80:
                      v45 = *(_QWORD *)(a3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                      if ( v45 == a3 + 48 )
                      {
                        v46 = 0;
                      }
                      else
                      {
                        if ( !v45 )
                          BUG();
                        v46 = v45 - 24;
                        if ( (unsigned int)*(unsigned __int8 *)(v45 - 24) - 30 >= 0xB )
                          v46 = 0;
                      }
                      LOWORD(v29) = 0;
                      sub_B444E0(v31, v46 + 24, v29);
                      sub_AE8F80(v31);
                    }
                    v26 = v66;
                    if ( v6 == v28 )
                      goto LABEL_38;
                    continue;
                  }
                }
              }
              else if ( sub_C8CA60((__int64)&v62, (__int64)v41) )
              {
                v9 = v6 - 24;
                v26 = v66;
                goto LABEL_70;
              }
            }
LABEL_74:
            ;
          }
        }
        v44 = __OFADD__(v55, v54);
        v54 += v55;
        if ( !v44 )
        {
          if ( (unsigned int)qword_50072E8 < v54 )
            goto LABEL_77;
          v6 = *(_QWORD *)(v6 + 8);
          if ( v58 != v6 )
            continue;
          goto LABEL_30;
        }
        if ( v55 <= 0 )
        {
          v6 = *(_QWORD *)(v6 + 8);
          v54 = 0x8000000000000000LL;
          if ( v58 != v6 )
            continue;
          goto LABEL_30;
        }
LABEL_77:
        v26 = v66;
LABEL_78:
        v34 = 0;
LABEL_39:
        if ( !v26 )
          _libc_free((unsigned __int64)v63);
        return v34;
      default:
        goto LABEL_20;
    }
  }
}
