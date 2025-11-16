// Function: sub_8848A0
// Address: 0x8848a0
//
_QWORD *__fastcall sub_8848A0(__int64 a1, __int64 *a2, unsigned int a3, int a4)
{
  _QWORD *v4; // r13
  __int64 v5; // r15
  char v6; // al
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // r13
  char v12; // al
  __int64 v13; // rsi
  __int64 v14; // r14
  char v15; // al
  _QWORD *v16; // r12
  __int64 v18; // rax
  __int64 i; // rbx
  __int64 **v20; // rax
  _QWORD *v21; // rax
  _QWORD *v22; // r8
  _QWORD *v23; // r14
  __int64 v24; // r15
  _QWORD *v25; // r14
  __int64 v26; // rbx
  __int64 v27; // rdi
  __int64 v28; // r12
  char v29; // si
  __int64 *v30; // r10
  __int64 v31; // r9
  char v32; // al
  char v33; // cl
  __int64 v34; // rdx
  char v35; // r11
  unsigned __int64 v36; // rcx
  __int64 **v37; // rdi
  __int64 **v38; // rdi
  __int64 v39; // rax
  __int64 *v40; // r11
  __int64 *v41; // rdx
  __int64 *v42; // r9
  __int64 *v43; // rax
  __int64 *v44; // rcx
  __int64 v45; // rdi
  __int64 v46; // r12
  __int64 v47; // rdx
  __int64 v48; // rsi
  __int64 v49; // rax
  __int64 v50; // rdi
  __int64 v51; // rsi
  __int64 v52; // rax
  char v53; // dl
  __int64 v54; // rax
  char v55; // al
  __int64 *v56; // rax
  __int64 *v57; // rdx
  __int64 *v58; // rax
  __int64 *v59; // rdx
  _QWORD *v60; // rax
  _QWORD *v61; // rdx
  __int64 v62; // rbx
  unsigned __int8 v63; // al
  __int64 v64; // rbx
  unsigned __int8 v65; // r12
  unsigned __int8 v66; // al
  __int64 *v67; // [rsp+8h] [rbp-88h]
  __int64 *v68; // [rsp+10h] [rbp-80h]
  __int64 *v69; // [rsp+10h] [rbp-80h]
  __int64 *v70; // [rsp+18h] [rbp-78h]
  _QWORD *v71; // [rsp+20h] [rbp-70h]
  _QWORD *v72; // [rsp+20h] [rbp-70h]
  int v76; // [rsp+40h] [rbp-50h]
  _QWORD *v78; // [rsp+48h] [rbp-48h]
  __int64 v79; // [rsp+50h] [rbp-40h]
  _QWORD *v80; // [rsp+50h] [rbp-40h]
  bool v81; // [rsp+58h] [rbp-38h]
  __int64 v82; // [rsp+58h] [rbp-38h]
  _QWORD *v83; // [rsp+58h] [rbp-38h]

  v4 = 0;
  v5 = **(_QWORD **)(a1 + 168);
  if ( !v5 )
    return v4;
  v76 = a3 & 2;
  do
  {
    while ( 2 )
    {
      while ( 2 )
      {
        v6 = *(_BYTE *)(v5 + 96);
        if ( dword_4D047C0 | dword_4D047C8 && !a4 && (v6 & 0x10) != 0
          || (v6 & 1) == 0
          || (v6 & 2) != 0 && ((_BYTE)sub_72B780(v5)[3] & 1) == 0 )
        {
          goto LABEL_38;
        }
        v7 = *(_QWORD *)(v5 + 40);
        v8 = *(_QWORD *)(v7 + 168);
        v81 = *(_BYTE *)(v7 + 140) == 9 && (*(_BYTE *)(v8 + 109) & 0x20) != 0;
        v9 = *(_QWORD *)(v8 + 152);
        if ( !v9 )
          goto LABEL_41;
        if ( (*(_BYTE *)(v9 + 29) & 0x20) != 0 )
          goto LABEL_41;
        v10 = sub_883800(*(_QWORD *)(*(_QWORD *)v7 + 96LL) + 192LL, *a2);
        if ( !v10 )
          goto LABEL_41;
        v71 = v4;
        v11 = v10;
        v79 = 0;
        do
        {
          while ( 1 )
          {
            if ( *(_DWORD *)(v11 + 40) != *(_DWORD *)(v9 + 24) )
              goto LABEL_15;
            if ( *(_BYTE *)(v11 + 80) == 13 )
              goto LABEL_15;
            if ( !sub_7D1FF0(v11, a3) )
              goto LABEL_15;
            v12 = *(_BYTE *)(v11 + 80);
            if ( v81 && v12 == 8 )
              goto LABEL_15;
            v13 = v11;
            if ( v12 == 16 )
            {
              v13 = **(_QWORD **)(v11 + 88);
              v12 = *(_BYTE *)(v13 + 80);
            }
            if ( v12 == 24 )
            {
              v13 = *(_QWORD *)(v13 + 88);
              v12 = *(_BYTE *)(v13 + 80);
            }
            if ( (unsigned __int8)(v12 - 4) <= 2u || v12 == 3 && *(_BYTE *)(v13 + 104) )
              break;
            if ( !v76 )
            {
LABEL_143:
              v14 = v11;
              v4 = v71;
              if ( (unsigned __int8)sub_877F80(v14) == 2 )
                goto LABEL_41;
              if ( (*(_BYTE *)(v14 + 82) & 4) == 0 )
                goto LABEL_30;
              goto LABEL_40;
            }
LABEL_15:
            v11 = *(_QWORD *)(v11 + 32);
            if ( !v11 )
              goto LABEL_28;
          }
          if ( v76 )
            goto LABEL_143;
          v79 = v11;
          v11 = *(_QWORD *)(v11 + 32);
        }
        while ( v11 );
LABEL_28:
        v14 = v79;
        v4 = v71;
        if ( !v79 )
          goto LABEL_41;
        if ( (*(_BYTE *)(v79 + 82) & 4) == 0 )
        {
LABEL_30:
          v82 = qword_4F5FE50;
          if ( qword_4F5FE50 )
            qword_4F5FE50 = *(_QWORD *)qword_4F5FE50;
          else
            v82 = sub_823970(32);
          *(_QWORD *)v82 = 0;
          *(_QWORD *)(v82 + 16) = 0;
          *(_BYTE *)(v82 + 24) = 0;
          *(_QWORD *)(v82 + 8) = v14;
          v15 = *(_BYTE *)(v14 + 80);
          if ( v15 == 17 )
          {
            v62 = *(_QWORD *)(v14 + 88);
            v63 = sub_87D550(v62);
            v64 = *(_QWORD *)(v62 + 8);
            v65 = v63;
            while ( v64 )
            {
              v66 = sub_87D550(v64);
              v64 = *(_QWORD *)(v64 + 8);
              if ( v65 > v66 )
                v65 = v66;
            }
            *(_BYTE *)(v82 + 24) = v65;
            v16 = *(_QWORD **)(v82 + 16);
          }
          else if ( v15 == 16 )
          {
            v16 = 0;
            *(_BYTE *)(v82 + 24) = *(_BYTE *)(v14 + 96) & 3;
          }
          else
          {
            *(_BYTE *)(v82 + 24) = sub_87D550(v14);
            v16 = *(_QWORD **)(v82 + 16);
          }
          v14 = 0;
          goto LABEL_44;
        }
LABEL_40:
        if ( *(_BYTE *)(v14 + 80) == 16 && (*(_BYTE *)(v14 + 96) & 4) != 0 )
          goto LABEL_42;
LABEL_41:
        v14 = 0;
LABEL_42:
        v18 = sub_8848A0(v7, a2, a3, 1);
        v82 = v18;
        if ( !v18 )
          goto LABEL_38;
        v16 = *(_QWORD **)(v18 + 16);
LABEL_44:
        for ( i = v82; ; v16 = *(_QWORD **)(i + 16) )
        {
          if ( v16 )
          {
            if ( !*v16 || (*(_BYTE *)(v16[2] + 96LL) & 2) == 0 )
            {
              v21 = sub_5EBAE0(v5, (__int64)v16);
              *(_QWORD *)(i + 16) = v21;
              v16[1] = v21;
            }
          }
          else
          {
            *(_QWORD *)(i + 16) = sub_5EBAE0(v5, 0);
          }
          if ( v14 )
            *(_BYTE *)(i + 24) = *(_BYTE *)(v14 + 96) & 3;
          v20 = (*(_BYTE *)(v5 + 96) & 2) != 0 ? sub_72B780(v5) : *(__int64 ***)(v5 + 112);
          *(_BYTE *)(i + 24) = sub_87D600(*(_BYTE *)(i + 24), *((unsigned __int8 *)v20 + 25));
          i = *(_QWORD *)i;
          if ( !i )
            break;
        }
        if ( !v4 )
        {
          v4 = (_QWORD *)v82;
          goto LABEL_38;
        }
        v22 = (_QWORD *)v82;
        v78 = 0;
        v70 = (__int64 *)v5;
        v72 = (_QWORD *)v82;
        while ( 2 )
        {
          while ( 1 )
          {
            v23 = v22;
            v22 = (_QWORD *)*v22;
            if ( v4 )
              break;
LABEL_86:
            v78 = v23;
            if ( !v22 )
              goto LABEL_87;
          }
          v83 = v4;
          v80 = v22;
          v24 = (__int64)v23;
          v25 = 0;
          while ( 2 )
          {
            while ( 2 )
            {
              v26 = (__int64)v4;
              v4 = (_QWORD *)*v4;
              v27 = *(_QWORD *)(v26 + 8);
              if ( (*(_BYTE *)(v27 + 82) & 4) != 0 || (v28 = *(_QWORD *)(v24 + 8), (*(_BYTE *)(v28 + 82) & 4) != 0) )
              {
LABEL_61:
                v25 = (_QWORD *)v26;
                if ( !v4 )
                  goto LABEL_85;
                continue;
              }
              break;
            }
            v29 = *(_BYTE *)(v27 + 80);
            v30 = *(__int64 **)(v24 + 16);
            v31 = *(_QWORD *)(v26 + 8);
            v32 = v29;
            if ( v29 == 16 )
            {
              v31 = **(_QWORD **)(v27 + 88);
              v32 = *(_BYTE *)(v31 + 80);
            }
            if ( v32 == 24 )
            {
              v31 = *(_QWORD *)(v31 + 88);
              v32 = *(_BYTE *)(v31 + 80);
            }
            v33 = *(_BYTE *)(v28 + 80);
            v34 = *(_QWORD *)(v24 + 8);
            v35 = v33;
            if ( v33 == 16 )
            {
              v34 = **(_QWORD **)(v28 + 88);
              v35 = *(_BYTE *)(v34 + 80);
            }
            if ( v35 == 24 )
              v34 = *(_QWORD *)(v34 + 88);
            if ( v34 == v31 )
            {
              switch ( v32 )
              {
                case 10:
                  v39 = *(_QWORD *)(*(_QWORD *)(v34 + 88) + 152LL);
                  break;
                case 17:
                  if ( *(_BYTE *)(v34 + 96) )
                    goto LABEL_105;
                  v52 = *(_QWORD *)(v34 + 88);
                  v53 = *(_BYTE *)(v52 + 80);
                  if ( v53 == 16 )
                  {
                    v52 = **(_QWORD **)(v52 + 88);
                    v53 = *(_BYTE *)(v52 + 80);
                  }
                  if ( v53 == 24 )
                  {
                    v52 = *(_QWORD *)(v52 + 88);
                    v53 = *(_BYTE *)(v52 + 80);
                  }
                  v54 = *(_QWORD *)(v52 + 88);
                  if ( v53 != 20 )
                  {
                    v39 = *(_QWORD *)(v54 + 152);
                    if ( *(_BYTE *)(v39 + 140) == 12 )
                    {
                      do
                        v39 = *(_QWORD *)(v39 + 160);
                      while ( *(_BYTE *)(v39 + 140) == 12 );
                      if ( !*(_QWORD *)(*(_QWORD *)(v39 + 168) + 40LL) )
                        goto LABEL_94;
LABEL_105:
                      v40 = *(__int64 **)(v26 + 16);
                      v41 = v40;
                      if ( v29 == 16 )
                      {
                        v69 = *(__int64 **)(v24 + 16);
                        v58 = sub_8778E0(*(_QWORD *)(v27 + 88), v40[2]);
                        v33 = *(_BYTE *)(v28 + 80);
                        v30 = v69;
                        v40 = v58;
                        v42 = v59;
                      }
                      else
                      {
                        do
                        {
                          v42 = v41;
                          v41 = (__int64 *)*v41;
                        }
                        while ( v41 );
                      }
                      v43 = v30;
                      if ( v33 == 16 )
                      {
                        v67 = v40;
                        v68 = v42;
                        v56 = sub_8778E0(*(_QWORD *)(v28 + 88), v30[2]);
                        v42 = v68;
                        v40 = v67;
                        v30 = v56;
                        v44 = v57;
                      }
                      else
                      {
                        do
                        {
                          v44 = v43;
                          v43 = (__int64 *)*v43;
                        }
                        while ( v43 );
                      }
                      v45 = v42[2];
                      v46 = v44[2];
                      v47 = *(_QWORD *)(v45 + 40);
                      v48 = *(_QWORD *)(v46 + 40);
                      if ( v47 == v48
                        || v47
                        && v48
                        && dword_4F07588
                        && (v49 = *(_QWORD *)(v47 + 32), v49 == *(_QWORD *)(v48 + 32))
                        && v49 )
                      {
                        v55 = *(_BYTE *)(v46 + 96) & 2;
                        if ( (*(_BYTE *)(v45 + 96) & 2) != 0 )
                        {
                          if ( v55 )
                            goto LABEL_94;
                          v30 = *(__int64 **)(v24 + 16);
                          v28 = *(_QWORD *)(v24 + 8);
                          v27 = *(_QWORD *)(v26 + 8);
                          goto LABEL_78;
                        }
                        if ( !v55 && (unsigned int)sub_5ED650(v40, v42, v30, v44) )
                          goto LABEL_94;
                      }
                      v30 = *(__int64 **)(v24 + 16);
                      v28 = *(_QWORD *)(v24 + 8);
                      v27 = *(_QWORD *)(v26 + 8);
LABEL_78:
                      if ( (unsigned int)sub_877730(v27, v28, v30, a1) )
                        goto LABEL_95;
                      if ( !(unsigned int)sub_877730(
                                            *(_QWORD *)(v24 + 8),
                                            *(_QWORD *)(v26 + 8),
                                            *(_QWORD **)(v26 + 16),
                                            a1) )
                        goto LABEL_61;
LABEL_80:
                      if ( v25 )
                        *v25 = v4;
                      else
                        v83 = v4;
                      v37 = *(__int64 ***)(v26 + 16);
                      if ( v37 )
                        sub_5EBA80(v37);
                      *(_QWORD *)v26 = qword_4F5FE50;
                      qword_4F5FE50 = v26;
                      if ( !v4 )
                      {
LABEL_85:
                        v22 = v80;
                        v4 = v83;
                        v23 = (_QWORD *)v24;
                        goto LABEL_86;
                      }
                      continue;
                    }
LABEL_104:
                    if ( !*(_QWORD *)(*(_QWORD *)(v39 + 168) + 40LL) )
                      goto LABEL_94;
                    goto LABEL_105;
                  }
                  v39 = *(_QWORD *)(*(_QWORD *)(v54 + 176) + 152LL);
                  break;
                case 8:
                  goto LABEL_105;
                default:
                  goto LABEL_94;
              }
              while ( *(_BYTE *)(v39 + 140) == 12 )
                v39 = *(_QWORD *)(v39 + 160);
              goto LABEL_104;
            }
            break;
          }
          v36 = *(unsigned __int8 *)(v34 + 80);
          if ( v32 != 3 || !*(_BYTE *)(v31 + 104) )
          {
            if ( (_BYTE)v36 == 3 && *(_BYTE *)(v34 + 104) )
            {
              if ( v32 == 3 )
                goto LABEL_119;
              v36 = (unsigned __int64)&dword_4F077C4;
              if ( dword_4F077C4 == 2 && (unsigned __int8)(v32 - 4) <= 2u )
              {
                v50 = *(_QWORD *)(v31 + 88);
                v51 = *(_QWORD *)(v34 + 88);
                if ( v50 == v51 )
                  goto LABEL_94;
LABEL_120:
                if ( (unsigned int)sub_8D97D0(v50, v51, 0, v36, v22) )
                  goto LABEL_94;
                v30 = *(__int64 **)(v24 + 16);
                v28 = *(_QWORD *)(v24 + 8);
                v27 = *(_QWORD *)(v26 + 8);
              }
            }
            goto LABEL_78;
          }
          if ( (_BYTE)v36 == 3 )
          {
            if ( !*(_BYTE *)(v34 + 104) )
              goto LABEL_119;
            goto LABEL_78;
          }
          if ( dword_4F077C4 != 2 )
            goto LABEL_78;
          v36 = (unsigned int)(v36 - 4);
          if ( (unsigned __int8)v36 > 2u )
            goto LABEL_78;
LABEL_119:
          v50 = *(_QWORD *)(v31 + 88);
          v51 = *(_QWORD *)(v34 + 88);
          if ( v50 != v51 )
            goto LABEL_120;
LABEL_94:
          if ( *(_BYTE *)(v24 + 24) < *(_BYTE *)(v26 + 24) )
            goto LABEL_80;
LABEL_95:
          v22 = v80;
          if ( v78 )
            *v78 = v80;
          else
            v72 = v80;
          v38 = *(__int64 ***)(v24 + 16);
          if ( v38 )
          {
            sub_5EBA80(v38);
            v22 = v80;
          }
          v4 = v83;
          *(_QWORD *)v24 = qword_4F5FE50;
          qword_4F5FE50 = v24;
          if ( v22 )
            continue;
          break;
        }
LABEL_87:
        v5 = (__int64)v70;
        if ( !v72 )
        {
LABEL_38:
          v5 = *(_QWORD *)v5;
          if ( !v5 )
            return v4;
          continue;
        }
        break;
      }
      if ( !v4 )
      {
        v5 = *v70;
        v4 = v72;
        if ( !*v70 )
          return v4;
        continue;
      }
      break;
    }
    v60 = v4;
    do
    {
      v61 = v60;
      v60 = (_QWORD *)*v60;
    }
    while ( v60 );
    *v61 = v72;
    v5 = *v70;
  }
  while ( *v70 );
  return v4;
}
