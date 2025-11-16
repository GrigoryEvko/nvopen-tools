// Function: sub_AE9DC0
// Address: 0xae9dc0
//
void __fastcall sub_AE9DC0(_QWORD *a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 *v5; // r12
  __int64 **v6; // rax
  __int64 v7; // rsi
  _QWORD *v8; // r14
  char v9; // al
  _QWORD *v10; // r12
  __int64 v11; // r13
  char v12; // bl
  __int64 v13; // rax
  unsigned int v14; // edx
  __int64 v15; // r15
  __int64 v16; // rcx
  __int64 *v17; // rbx
  unsigned __int64 v18; // r15
  unsigned __int64 v19; // r9
  bool v20; // r13
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // r10
  __int64 v26; // rax
  __int64 v27; // rax
  unsigned __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rdx
  __int64 v31; // r15
  __int64 v32; // rax
  unsigned int v33; // edx
  bool v34; // zf
  __int64 v35; // rax
  __int64 v36; // rax
  int v37; // r8d
  __int64 v38; // r13
  __int64 *v39; // r12
  __int64 **v40; // rax
  int v41; // eax
  __int64 v42; // [rsp-8h] [rbp-2A8h]
  __int64 v43; // [rsp+8h] [rbp-298h]
  _QWORD *v45; // [rsp+18h] [rbp-288h]
  __int64 v47; // [rsp+28h] [rbp-278h]
  bool v48; // [rsp+37h] [rbp-269h]
  unsigned __int64 v50; // [rsp+58h] [rbp-248h]
  __int64 *v51; // [rsp+60h] [rbp-240h]
  __int64 v52; // [rsp+68h] [rbp-238h]
  __int64 v53; // [rsp+70h] [rbp-230h]
  __int64 v54; // [rsp+70h] [rbp-230h]
  int v55; // [rsp+78h] [rbp-228h]
  __int64 v56; // [rsp+78h] [rbp-228h]
  __int64 v57; // [rsp+90h] [rbp-210h] BYREF
  unsigned __int64 v58; // [rsp+98h] [rbp-208h]
  __int64 v59; // [rsp+A0h] [rbp-200h]
  bool v60; // [rsp+A8h] [rbp-1F8h]
  char v61; // [rsp+B0h] [rbp-1F0h]
  _QWORD v62[60]; // [rsp+C0h] [rbp-1E0h] BYREF

  if ( *(_DWORD *)(a3 + 16) )
  {
    v45 = a1;
    if ( a1 )
    {
      v43 = sub_AA48A0((__int64)(a1 - 3));
      v5 = (__int64 *)sub_AA4B30((__int64)(a1 - 3));
      v6 = (__int64 **)sub_BCB2A0(v43);
      v7 = (__int64)v5;
      v47 = sub_ACADE0(v6);
      sub_AE0470((__int64)v62, v5, 0, 0);
      if ( a1 != a2 )
      {
        while ( 1 )
        {
          v8 = (_QWORD *)v45[4];
          if ( v45 + 3 != v8 )
            break;
LABEL_31:
          v45 = (_QWORD *)v45[1];
          if ( v45 == a2 )
            goto LABEL_56;
          if ( !v45 )
            goto LABEL_66;
        }
        while ( 1 )
        {
          if ( !v8 )
            BUG();
          v9 = *((_BYTE *)v8 - 24);
          v10 = v8 - 3;
          switch ( v9 )
          {
            case '<':
              v7 = a4;
              sub_AE9D60((__int64)&v57, a4, (__int64)(v8 - 3));
              v52 = (__int64)(v8 - 3);
              v11 = v57;
              v12 = v61;
              v53 = v47;
              break;
            case '>':
              v7 = a4;
              sub_AE9D00((__int64)&v57, a4, (__int64)(v8 - 3));
              v11 = v57;
              v12 = v61;
              v53 = *(v8 - 11);
              v52 = *(v8 - 7);
              break;
            case 'U':
              v30 = *(v8 - 7);
              if ( !v30 )
                goto LABEL_30;
              if ( !*(_BYTE *)v30
                && *(_QWORD *)(v30 + 24) == v8[7]
                && (*(_BYTE *)(v30 + 33) & 0x20) != 0
                && ((v41 = *(_DWORD *)(v30 + 36), v41 == 238) || (unsigned int)(v41 - 240) <= 1) )
              {
                v7 = a4;
                sub_AE9C80((__int64)&v57, a4, (__int64)(v8 - 3));
                v11 = v57;
                v12 = v61;
                v52 = v8[-4 * (*((_DWORD *)v8 - 5) & 0x7FFFFFF) - 3];
                v53 = v47;
              }
              else
              {
                if ( *(_BYTE *)v30
                  || *(_QWORD *)(v30 + 24) != v8[7]
                  || (*(_BYTE *)(v30 + 33) & 0x20) == 0
                  || ((*(_DWORD *)(v30 + 36) - 243) & 0xFFFFFFFD) != 0 )
                {
                  goto LABEL_30;
                }
                v7 = a4;
                sub_AE9C80((__int64)&v57, a4, (__int64)(v8 - 3));
                v11 = v57;
                v12 = v61;
                v31 = *((_DWORD *)v8 - 5) & 0x7FFFFFF;
                v32 = v10[4 * (1 - v31)];
                v54 = v32;
                if ( *(_BYTE *)v32 == 17 )
                {
                  v33 = *(_DWORD *)(v32 + 32);
                  if ( v33 <= 0x40 )
                  {
                    if ( *(_QWORD *)(v32 + 24) )
                      v32 = v47;
                    v53 = v32;
                  }
                  else
                  {
                    v34 = v33 == (unsigned int)sub_C444A0(v32 + 24);
                    v35 = v54;
                    if ( !v34 )
                      v35 = v47;
                    v53 = v35;
                  }
                }
                else
                {
                  v53 = v47;
                }
                v52 = v10[-4 * v31];
              }
              break;
            default:
              goto LABEL_30;
          }
          if ( v12 )
          {
            v7 = *(_QWORD *)(a3 + 8);
            v13 = *(unsigned int *)(a3 + 24);
            if ( (_DWORD)v13 )
            {
              v14 = (v13 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
              v15 = v7 + 88LL * v14;
              v16 = *(_QWORD *)v15;
              if ( v11 == *(_QWORD *)v15 )
              {
LABEL_11:
                if ( v15 != v7 + 88 * v13 )
                {
                  if ( (*((_BYTE *)v8 - 17) & 0x20) == 0 || (v7 = 38, !sub_B91C10(v8 - 3, 38)) )
                  {
                    v36 = sub_AF40E0(v43, 1, 1);
                    v7 = 38;
                    sub_B99FD0(v8 - 3, 38, v36);
                  }
                  v17 = *(__int64 **)(v15 + 40);
                  v51 = &v17[2 * *(unsigned int *)(v15 + 48)];
                  if ( v51 != v17 )
                  {
                    v18 = v58;
                    v48 = v60;
                    v50 = v58 + v59;
                    do
                    {
                      if ( (*((_BYTE *)v8 - 17) & 0x20) != 0 )
                      {
                        v7 = 38;
                        sub_B91C10(v8 - 3, 38);
                      }
                      v28 = sub_AF3FE0(*v17);
                      if ( (_BYTE)v29 )
                      {
                        v19 = v50;
                        if ( v50 > v28 )
                          v19 = v28;
                        if ( v19 <= v18 )
                          goto LABEL_24;
                        v20 = v19 >= v28 && v18 == 0;
                      }
                      else
                      {
                        LODWORD(v19) = v50;
                        v20 = v48;
                      }
                      v55 = v19;
                      v21 = sub_BD5C60(v8 - 3, v7, v29);
                      v22 = 0;
                      v23 = sub_B0D000(v21, 0, 0, 0, 1);
                      v25 = v23;
                      if ( !v20 )
                      {
                        v22 = (unsigned int)v18;
                        v25 = sub_B0E470(v23, (unsigned int)v18, (unsigned int)(v55 - v18));
                      }
                      v56 = v25;
                      v26 = sub_BD5C60(v8 - 3, v22, v24);
                      v27 = sub_B0D000(v26, 0, 0, 0, 1);
                      if ( *(_BYTE *)(v8[2] + 40LL) )
                      {
                        sub_B12940((_DWORD)v8 - 24, v53, *v17, v56, v52, v27, v17[1]);
                        v7 = v42;
                      }
                      else
                      {
                        v7 = (__int64)(v8 - 3);
                        sub_ADE690(v62, (__int64)(v8 - 3), v53, *v17, v56, v52, v27, v17[1]);
                      }
LABEL_24:
                      v17 += 2;
                    }
                    while ( v51 != v17 );
                  }
                }
              }
              else
              {
                v37 = 1;
                while ( v16 != -4096 )
                {
                  v14 = (v13 - 1) & (v37 + v14);
                  v15 = v7 + 88LL * v14;
                  v16 = *(_QWORD *)v15;
                  if ( v11 == *(_QWORD *)v15 )
                    goto LABEL_11;
                  ++v37;
                }
              }
            }
          }
LABEL_30:
          v8 = (_QWORD *)v8[1];
          if ( v45 + 3 == v8 )
            goto LABEL_31;
        }
      }
    }
    else
    {
      v38 = sub_AA48A0(0);
      v39 = (__int64 *)sub_AA4B30(0);
      v40 = (__int64 **)sub_BCB2A0(v38);
      sub_ACADE0(v40);
      v7 = (__int64)v39;
      sub_AE0470((__int64)v62, v39, 0, 0);
      if ( a2 )
LABEL_66:
        BUG();
    }
LABEL_56:
    sub_AE9130((__int64)v62, v7);
  }
}
