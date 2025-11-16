// Function: sub_139C7C0
// Address: 0x139c7c0
//
void __fastcall sub_139C7C0(__int64 a1, __int64 a2)
{
  __int64 v2; // rsi
  unsigned int v3; // r12d
  __int64 v4; // rdx
  _QWORD *v5; // rbx
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // r14
  __int64 v9; // rdx
  __int64 v10; // r15
  unsigned __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  char v19; // al
  unsigned __int8 v20; // al
  __int64 v21; // rdi
  __int64 v22; // rax
  unsigned __int64 v23; // rcx
  _QWORD *v24; // r15
  _QWORD *v25; // r14
  __int64 v26; // rbx
  __int64 v27; // r13
  _QWORD *v28; // r12
  unsigned int v29; // esi
  char v30; // al
  _QWORD *v31; // rax
  _QWORD *v32; // rdx
  _BYTE *v33; // rcx
  _QWORD *v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // rax
  __int64 v40; // rax
  int v41; // ecx
  _BYTE *v42; // rdx
  bool v43; // zf
  __int64 v44; // rax
  unsigned __int64 v45; // r15
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rcx
  __int64 v49; // rdx
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rdi
  unsigned int v61; // r14d
  __int64 v62; // [rsp+8h] [rbp-2C8h]
  char v63; // [rsp+8h] [rbp-2C8h]
  unsigned __int64 v64; // [rsp+18h] [rbp-2B8h]
  _QWORD *v65; // [rsp+18h] [rbp-2B8h]
  __int64 v66; // [rsp+20h] [rbp-2B0h]
  char v67; // [rsp+20h] [rbp-2B0h]
  __int64 v68; // [rsp+20h] [rbp-2B0h]
  __int64 v69; // [rsp+20h] [rbp-2B0h]
  __int64 v70; // [rsp+28h] [rbp-2A8h] BYREF
  __int64 v71; // [rsp+38h] [rbp-298h] BYREF
  __int64 v72[4]; // [rsp+40h] [rbp-290h] BYREF
  _BYTE *v73; // [rsp+60h] [rbp-270h] BYREF
  __int64 v74; // [rsp+68h] [rbp-268h]
  _BYTE v75[256]; // [rsp+70h] [rbp-260h] BYREF
  __int64 v76; // [rsp+170h] [rbp-160h] BYREF
  _BYTE *v77; // [rsp+178h] [rbp-158h]
  _BYTE *v78; // [rsp+180h] [rbp-150h]
  __int64 v79; // [rsp+188h] [rbp-148h]
  int v80; // [rsp+190h] [rbp-140h]
  _BYTE v81[312]; // [rsp+198h] [rbp-138h] BYREF

  v73 = v75;
  v74 = 0x2000000000LL;
  v77 = v81;
  v78 = v81;
  v70 = a2;
  v2 = *(_QWORD *)(a1 + 8);
  v72[1] = (__int64)&v70;
  v76 = 0;
  v79 = 32;
  v80 = 0;
  v72[0] = (__int64)&v76;
  v72[2] = (__int64)&v73;
  sub_139C140(v72, v2);
  v3 = v74;
  if ( !(_DWORD)v74 )
    goto LABEL_9;
  while ( 2 )
  {
    v4 = v3--;
    v5 = *(_QWORD **)&v73[8 * v4 - 8];
    LODWORD(v74) = v3;
    v6 = sub_1648700(v5);
    v7 = *v5;
    v8 = v6;
    v9 = *(unsigned __int8 *)(v6 + 16);
    switch ( *(_BYTE *)(v6 + 16) )
    {
      case 0x1D:
      case 0x4E:
        if ( (unsigned __int8)v9 <= 0x17u )
        {
          v64 = 0;
          v11 = 0;
          goto LABEL_18;
        }
        v10 = v6 | 4;
        if ( (_BYTE)v9 == 78 )
          goto LABEL_74;
        v64 = 0;
        v11 = 0;
        if ( (_BYTE)v9 == 29 )
        {
          v10 = v6 & 0xFFFFFFFFFFFFFFFBLL;
LABEL_74:
          v43 = ((v10 >> 2) & 1) == 0;
          v44 = (v10 >> 2) & 1;
          v64 = v10 & 0xFFFFFFFFFFFFFFF8LL;
          v11 = v10 & 0xFFFFFFFFFFFFFFF8LL;
          v45 = (v10 & 0xFFFFFFFFFFFFFFF8LL) + 56;
          v63 = v44;
          if ( !v43 )
          {
            if ( !(unsigned __int8)sub_1560260(v45, 0xFFFFFFFFLL, 36) )
            {
              if ( *(char *)(v64 + 23) < 0 )
              {
                v46 = sub_1648A40(v64);
                v48 = v46 + v47;
                v49 = 0;
                if ( *(char *)(v64 + 23) < 0 )
                {
                  v68 = v48;
                  v50 = sub_1648A40(v64);
                  v48 = v68;
                  v49 = v50;
                }
                if ( (unsigned int)((v48 - v49) >> 4) )
                  goto LABEL_119;
              }
              v51 = *(_QWORD *)(v64 - 24);
              if ( *(_BYTE *)(v51 + 16)
                || (v71 = *(_QWORD *)(v51 + 112), !(unsigned __int8)sub_1560260(&v71, 0xFFFFFFFFLL, 36)) )
              {
LABEL_119:
                if ( !(unsigned __int8)sub_1560260(v45, 0xFFFFFFFFLL, 37) )
                {
                  if ( *(char *)(v64 + 23) < 0 )
                  {
                    v53 = sub_1648A40(v64);
                    v55 = v53 + v54;
                    if ( *(char *)(v64 + 23) >= 0 )
                    {
                      v56 = 0;
                    }
                    else
                    {
                      v69 = v53 + v54;
                      v56 = sub_1648A40(v64);
                      v55 = v69;
                    }
                    if ( v56 != v55 )
                    {
                      while ( *(_DWORD *)(*(_QWORD *)v56 + 8LL) <= 1u )
                      {
                        v56 += 16;
                        if ( v55 == v56 )
                          goto LABEL_92;
                      }
                      goto LABEL_94;
                    }
                  }
LABEL_92:
                  v57 = *(_QWORD *)(v64 - 24);
                  if ( *(_BYTE *)(v57 + 16) )
                    goto LABEL_94;
                  v71 = *(_QWORD *)(v57 + 112);
                  if ( !(unsigned __int8)sub_1560260(&v71, 0xFFFFFFFFLL, 37) )
                    goto LABEL_94;
                }
              }
            }
            v67 = sub_1560260(v45, 0xFFFFFFFFLL, 30);
            if ( v67 )
              goto LABEL_98;
            v52 = *(_QWORD *)(v64 - 24);
            if ( !*(_BYTE *)(v52 + 16) )
            {
              v71 = *(_QWORD *)(v52 + 112);
              v19 = sub_1560260(&v71, 0xFFFFFFFFLL, 30);
              v67 = v63;
              goto LABEL_28;
            }
LABEL_94:
            v67 = v63;
            goto LABEL_29;
          }
        }
LABEL_18:
        if ( (unsigned __int8)sub_1560260(v11 + 56, 0xFFFFFFFFLL, 36) )
          goto LABEL_25;
        if ( *(char *)(v11 + 23) >= 0 )
          goto LABEL_120;
        v12 = sub_1648A40(v11);
        v14 = v12 + v13;
        v15 = 0;
        if ( *(char *)(v11 + 23) < 0 )
        {
          v66 = v14;
          v16 = sub_1648A40(v11);
          v14 = v66;
          v15 = v16;
        }
        if ( !(unsigned int)((v14 - v15) >> 4) )
        {
LABEL_120:
          v17 = *(_QWORD *)(v11 - 72);
          if ( !*(_BYTE *)(v17 + 16) )
          {
            v71 = *(_QWORD *)(v17 + 112);
            if ( (unsigned __int8)sub_1560260(&v71, 0xFFFFFFFFLL, 36) )
              goto LABEL_25;
          }
        }
        v67 = sub_1560260(v11 + 56, 0xFFFFFFFFLL, 37);
        if ( v67 )
          goto LABEL_25;
        if ( *(char *)(v11 + 23) < 0 )
        {
          v36 = sub_1648A40(v11);
          v38 = v36 + v37;
          if ( *(char *)(v11 + 23) >= 0 )
          {
            v39 = 0;
          }
          else
          {
            v62 = v36 + v37;
            v39 = sub_1648A40(v11);
            v38 = v62;
          }
          for ( ; v38 != v39; v39 += 16 )
          {
            if ( *(_DWORD *)(*(_QWORD *)v39 + 8LL) > 1u )
              goto LABEL_29;
          }
        }
        v40 = *(_QWORD *)(v11 - 72);
        if ( !*(_BYTE *)(v40 + 16) )
        {
          v71 = *(_QWORD *)(v40 + 112);
          v67 = sub_1560260(&v71, 0xFFFFFFFFLL, 37);
          if ( v67 )
          {
LABEL_25:
            v67 = sub_1560260(v11 + 56, 0xFFFFFFFFLL, 30);
            if ( !v67 )
            {
              v18 = *(_QWORD *)(v11 - 72);
              if ( *(_BYTE *)(v18 + 16) )
                goto LABEL_29;
              v71 = *(_QWORD *)(v18 + 112);
              v19 = sub_1560260(&v71, 0xFFFFFFFFLL, 30);
LABEL_28:
              if ( !v19 )
                goto LABEL_29;
              goto LABEL_98;
            }
            v67 = 0;
LABEL_98:
            if ( !*(_BYTE *)(*(_QWORD *)v8 + 8LL) )
              goto LABEL_7;
          }
        }
LABEL_29:
        v20 = *(_BYTE *)(v11 + 16);
        v21 = 0;
        if ( v20 > 0x17u )
        {
          if ( v20 == 78 )
          {
            v21 = v64 | 4;
          }
          else
          {
            v21 = 0;
            if ( v20 == 29 )
              v21 = v64;
          }
        }
        if ( (unsigned __int8)sub_14AD0D0(v21) )
        {
          sub_139C140(v72, *(_QWORD *)(v8 + 8));
          goto LABEL_7;
        }
        if ( *(_BYTE *)(v8 + 16) == 78 )
        {
          v59 = *(_QWORD *)(v8 - 24);
          if ( !*(_BYTE *)(v59 + 16)
            && (*(_BYTE *)(v59 + 33) & 0x20) != 0
            && (unsigned int)(*(_DWORD *)(v59 + 36) - 133) <= 4
            && ((1LL << (*(_BYTE *)(v59 + 36) + 123)) & 0x15) != 0 )
          {
            v60 = *(_QWORD *)(v8 + 24 * (3LL - (*(_DWORD *)(v8 + 20) & 0xFFFFFFF)));
            v61 = *(_DWORD *)(v60 + 32);
            if ( v61 <= 0x40 )
            {
              if ( *(_QWORD *)(v60 + 24) )
              {
LABEL_110:
                if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)v70 + 32LL))(v70, v5) )
                  goto LABEL_9;
              }
            }
            else if ( v61 != (unsigned int)sub_16A57B0(v60 + 24) )
            {
              goto LABEL_110;
            }
          }
        }
        v22 = 24LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF);
        v65 = (_QWORD *)(v11 - v22);
        if ( (*(_BYTE *)(v11 + 23) & 0x40) != 0 )
          v65 = *(_QWORD **)(v11 - 8);
        v23 = (-(__int64)(v67 == 0) & 0xFFFFFFFFFFFFFFD0LL) + v22 - 24;
        if ( v65 == (_QWORD *)((char *)v65 + v23) )
          goto LABEL_7;
        v24 = v65;
        v25 = v5;
        v26 = v7;
        v27 = v11;
        v28 = (_QWORD *)((char *)v65 + v23);
        while ( 1 )
        {
          if ( v26 == *v24 )
          {
            v29 = -1431655765 * (v24 - v65) + 1;
            v30 = v67 ? sub_139C340(v27, v29, 22) : sub_139C570(v27, v29, 22);
            if ( !v30 && (*(unsigned __int8 (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)v70 + 32LL))(v70, v25) )
              break;
          }
          v24 += 3;
          if ( v24 == v28 )
            goto LABEL_7;
        }
LABEL_9:
        if ( v78 != v77 )
          _libc_free((unsigned __int64)v78);
        if ( v73 != v75 )
          _libc_free((unsigned __int64)v73);
        return;
      case 0x36:
        goto LABEL_5;
      case 0x37:
        if ( (*(_BYTE *)(v6 + 23) & 0x40) != 0 )
        {
          v31 = *(_QWORD **)(v6 - 8);
        }
        else
        {
          v9 = 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF);
          v31 = (_QWORD *)(v6 - v9);
        }
        if ( v7 == *v31 || (*(_BYTE *)(v8 + 18) & 1) != 0 )
          goto LABEL_6;
        goto LABEL_8;
      case 0x38:
      case 0x47:
      case 0x48:
      case 0x4D:
      case 0x4F:
        sub_139C140(v72, *(_QWORD *)(v6 + 8));
        v3 = v74;
        goto LABEL_8;
      case 0x3A:
        if ( v7 != *(_QWORD *)(v6 - 48) )
          goto LABEL_4;
        goto LABEL_6;
      case 0x3B:
LABEL_4:
        if ( v7 == *(_QWORD *)(v6 - 24) )
          goto LABEL_6;
LABEL_5:
        if ( (*(_BYTE *)(v6 + 18) & 1) != 0 )
          goto LABEL_6;
        goto LABEL_8;
      case 0x4B:
        if ( (*(_BYTE *)(v6 + 23) & 0x40) != 0 )
        {
          v32 = *(_QWORD **)(v6 - 8);
          v33 = (_BYTE *)v32[3];
          v34 = v32;
          if ( v33[16] != 15 )
            goto LABEL_55;
          if ( *(_DWORD *)(*(_QWORD *)v33 + 8LL) >> 8 )
            goto LABEL_54;
LABEL_101:
          v58 = sub_1649C60(*v5);
          if ( (unsigned __int8)sub_134E780(v58) )
            goto LABEL_7;
          if ( (*(_BYTE *)(v8 + 23) & 0x40) != 0 )
          {
            v32 = *(_QWORD **)(v8 - 8);
LABEL_54:
            v34 = v32;
            goto LABEL_55;
          }
          v41 = *(_DWORD *)(v8 + 20);
          goto LABEL_72;
        }
        v41 = *(_DWORD *)(v6 + 20);
        v34 = (_QWORD *)(v6 - 24LL * (v41 & 0xFFFFFFF));
        v42 = (_BYTE *)v34[3];
        if ( v42[16] != 15 )
          goto LABEL_55;
        if ( !(*(_DWORD *)(*(_QWORD *)v42 + 8LL) >> 8) )
          goto LABEL_101;
LABEL_72:
        v34 = (_QWORD *)(v8 - 24LL * (v41 & 0xFFFFFFF));
LABEL_55:
        v9 = 3LL * (*v34 == v7);
        v35 = v34[3 * (*v34 == v7)];
        if ( *(_BYTE *)(v35 + 16) != 54 || *(_BYTE *)(*(_QWORD *)(v35 - 24) + 16LL) != 3 )
        {
LABEL_6:
          if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD *, __int64))(*(_QWORD *)v70 + 32LL))(v70, v5, v9) )
            goto LABEL_9;
        }
LABEL_7:
        v3 = v74;
LABEL_8:
        if ( !v3 )
          goto LABEL_9;
        continue;
      case 0x52:
        goto LABEL_8;
      default:
        goto LABEL_6;
    }
  }
}
