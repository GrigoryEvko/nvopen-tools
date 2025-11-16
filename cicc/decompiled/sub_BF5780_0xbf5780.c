// Function: sub_BF5780
// Address: 0xbf5780
//
unsigned __int64 __fastcall sub_BF5780(__int64 **a1, _BYTE *a2, const char *a3, char a4)
{
  char *v5; // rdi
  unsigned __int8 v6; // dl
  int v7; // ebx
  __int64 v8; // rax
  __int64 v9; // rax
  unsigned int v10; // ebx
  int v11; // r14d
  char v12; // cl
  bool v13; // al
  __int64 v14; // rdx
  __int64 v15; // r13
  __int64 *v16; // r13
  __int64 v17; // r15
  _BYTE *v18; // rax
  __int64 v19; // rax
  __int64 v20; // r8
  __int64 v21; // rsi
  _BYTE **v22; // rdx
  __int64 *v23; // r13
  __int64 v24; // r15
  _BYTE *v25; // rax
  bool v26; // zf
  __int64 v27; // rcx
  __int64 v28; // rdx
  __int64 v29; // rdi
  _BYTE *v30; // rax
  char *v31; // rax
  unsigned __int64 result; // rax
  char v33; // r15
  __int64 v34; // r9
  int v35; // eax
  const char *v36; // r8
  __int64 v37; // rdx
  __int64 *v38; // r8
  __int64 v39; // r11
  _BYTE *v40; // rax
  __int64 v41; // rdi
  _BYTE *v42; // rax
  _BYTE *v43; // r8
  _BYTE *v44; // rax
  __int64 v45; // rdi
  _BYTE *v46; // rax
  __int64 *v47; // rdi
  const char *v48; // rax
  __int64 v49; // rdx
  __int64 *v50; // rdi
  char v51; // [rsp+8h] [rbp-C8h]
  __int64 v52; // [rsp+8h] [rbp-C8h]
  __int64 *v53; // [rsp+8h] [rbp-C8h]
  char v54; // [rsp+18h] [rbp-B8h]
  __int64 v55; // [rsp+18h] [rbp-B8h]
  __int64 v56; // [rsp+18h] [rbp-B8h]
  char v57; // [rsp+18h] [rbp-B8h]
  _BYTE *v58; // [rsp+18h] [rbp-B8h]
  __int64 v61; // [rsp+30h] [rbp-A0h]
  unsigned __int64 v62; // [rsp+30h] [rbp-A0h]
  const char *v63[2]; // [rsp+38h] [rbp-98h] BYREF
  _BYTE *v64; // [rsp+48h] [rbp-88h] BYREF
  __int128 v65; // [rsp+50h] [rbp-80h] BYREF
  __int64 v66; // [rsp+60h] [rbp-70h]
  _QWORD v67[4]; // [rsp+70h] [rbp-60h] BYREF
  char v68; // [rsp+90h] [rbp-40h]
  char v69; // [rsp+91h] [rbp-3Fh]

  v5 = (char *)a3;
  v63[0] = a3;
  v6 = *(a3 - 16);
  if ( (v6 & 2) != 0 )
  {
    v7 = *((_DWORD *)v5 - 6);
    if ( v7 != 2 )
      goto LABEL_3;
LABEL_46:
    if ( (unsigned __int8)sub_BF5560((__int64)a1, v5) )
      return 0;
    return 0xFFFFFFFF00000001LL;
  }
  v7 = (*((_WORD *)v5 - 8) >> 6) & 0xF;
  if ( v7 == 2 )
    goto LABEL_46;
LABEL_3:
  if ( !a4 )
  {
    v10 = v7 & 1;
    if ( !v10 )
    {
      v47 = *a1;
      if ( !*a1 )
        return 0xFFFFFFFF00000001LL;
      v69 = 1;
      v48 = "Struct tag nodes must have an odd number of operands!";
      goto LABEL_104;
    }
    if ( (v6 & 2) != 0 )
      v31 = (char *)*((_QWORD *)v5 - 4);
    else
      v31 = &v5[-8 * ((v6 >> 2) & 0xF) - 16];
    if ( !**(_BYTE **)v31 )
    {
      v66 = 0;
      v11 = 2;
      v65 = 0;
      goto LABEL_11;
    }
    v47 = *a1;
    if ( *a1 )
    {
      v69 = 1;
      v48 = "Struct tag nodes have a string as their first operand";
      goto LABEL_104;
    }
    return 0xFFFFFFFF00000001LL;
  }
  if ( (unsigned int)(-1431655765 * v7) > 0x55555555 )
  {
    v47 = *a1;
    if ( *a1 )
    {
      v69 = 1;
      v48 = "Access tag nodes must have the number of operands that is a multiple of 3!";
LABEL_104:
      v67[0] = v48;
      v68 = 3;
      sub_BE1BE0(v47, (__int64)v67, v63);
      return 0xFFFFFFFF00000001LL;
    }
    return 0xFFFFFFFF00000001LL;
  }
  if ( (v6 & 2) != 0 )
    v8 = *((_QWORD *)v5 - 4);
  else
    v8 = (__int64)&v5[-8 * ((v6 >> 2) & 0xF) - 16];
  v9 = *(_QWORD *)(v8 + 8);
  if ( !v9 || *(_BYTE *)v9 != 1 || **(_BYTE **)(v9 + 136) != 17 )
  {
    v50 = *a1;
    *(_QWORD *)&v65 = a2;
    if ( v50 )
    {
      v69 = 1;
      v67[0] = "Type size nodes must be constants!";
      v68 = 3;
      sub_BF02E0(v50, (__int64)v67, (_BYTE **)&v65, v63);
    }
    return 0xFFFFFFFF00000001LL;
  }
  v66 = 0;
  v10 = 3;
  v11 = 3;
  v65 = 0;
LABEL_11:
  LODWORD(v61) = -1;
  v12 = 0;
  v13 = (*(v5 - 16) & 2) != 0;
  while ( v13 )
  {
    if ( v10 >= *((_DWORD *)v5 - 6) )
      goto LABEL_82;
    v20 = *((_QWORD *)v5 - 4);
    v21 = v10 + 1;
    v22 = (_BYTE **)(v20 + 8LL * v10);
LABEL_30:
    if ( (unsigned __int8)(**v22 - 5) <= 0x1Fu )
    {
      v14 = *(_QWORD *)(v20 + 8 * v21);
      if ( v14 )
      {
        if ( *(_BYTE *)v14 == 1 )
        {
          v15 = *(_QWORD *)(v14 + 136);
          if ( *(_BYTE *)v15 == 17 )
          {
            if ( (_DWORD)v61 == -1 )
            {
              LODWORD(v61) = *(_DWORD *)(v15 + 32);
            }
            else if ( (_DWORD)v61 != *(_DWORD *)(v15 + 32) )
            {
              v16 = *a1;
              v12 = 1;
              if ( !*a1 )
                goto LABEL_26;
              v69 = 1;
              v67[0] = "Bitwidth between the offsets and struct type entries must match";
              v68 = 3;
              v17 = *v16;
              if ( !*v16 )
              {
                *((_BYTE *)v16 + 152) = 1;
                goto LABEL_25;
              }
              sub_CA0E80(v67, *v16);
              v18 = *(_BYTE **)(v17 + 32);
              if ( (unsigned __int64)v18 >= *(_QWORD *)(v17 + 24) )
              {
                sub_CB5D20(v17, 10);
              }
              else
              {
                *(_QWORD *)(v17 + 32) = v18 + 1;
                *v18 = 10;
              }
              v19 = *v16;
              *((_BYTE *)v16 + 152) = 1;
              if ( v19 )
              {
                sub_BDBD80((__int64)v16, a2);
                v5 = (char *)v63[0];
                if ( v63[0] )
                {
                  sub_BD9900(v16, v63[0]);
                  goto LABEL_24;
                }
                goto LABEL_25;
              }
              goto LABEL_24;
            }
            v33 = v66;
            v34 = v15 + 24;
            if ( (_BYTE)v66 )
            {
              v51 = v12;
              v35 = sub_C49970(&v65, v15 + 24);
              v34 = v15 + 24;
              v12 = v51;
              if ( v35 <= 0 )
                goto LABEL_57;
              v39 = (__int64)*a1;
              if ( *a1 )
              {
                v69 = 1;
                v67[0] = "Offsets must be increasing!";
                v68 = 3;
                if ( *(_QWORD *)v39 )
                {
                  v52 = v39;
                  v55 = *(_QWORD *)v39;
                  sub_CA0E80(v67, *(_QWORD *)v39);
                  v34 = v15 + 24;
                  v39 = v52;
                  v40 = *(_BYTE **)(v55 + 32);
                  if ( (unsigned __int64)v40 >= *(_QWORD *)(v55 + 24) )
                  {
                    sub_CB5D20(v55, 10);
                    v34 = v15 + 24;
                    v39 = v52;
                  }
                  else
                  {
                    *(_QWORD *)(v55 + 32) = v40 + 1;
                    *v40 = 10;
                  }
                }
                v26 = *(_QWORD *)v39 == 0;
                *(_BYTE *)(v39 + 152) = 1;
                if ( !v26 )
                {
                  v56 = v34;
                  v53 = (__int64 *)v39;
                  sub_BDBD80(v39, a2);
                  v34 = v56;
                  if ( v63[0] )
                  {
                    sub_A62C00(v63[0], *v53, (__int64)(v53 + 2), v53[1]);
                    v34 = v56;
                    v41 = *v53;
                    v42 = *(_BYTE **)(*v53 + 32);
                    if ( (unsigned __int64)v42 >= *(_QWORD *)(*v53 + 24) )
                    {
                      sub_CB5D20(v41, 10);
                      v34 = v56;
                    }
                    else
                    {
                      *(_QWORD *)(v41 + 32) = v42 + 1;
                      *v42 = 10;
                    }
                  }
                }
                v12 = v66;
                if ( !(_BYTE)v66 )
                {
                  v12 = v33;
                  goto LABEL_79;
                }
              }
              else
              {
                v12 = v33;
              }
LABEL_57:
              if ( DWORD2(v65) <= 0x40 && *(_DWORD *)(v15 + 32) <= 0x40u )
              {
                v49 = *(_QWORD *)(v15 + 24);
                DWORD2(v65) = *(_DWORD *)(v15 + 32);
                *(_QWORD *)&v65 = v49;
              }
              else
              {
                v54 = v12;
                sub_C43990(&v65, v34);
                v12 = v54;
              }
            }
            else
            {
LABEL_79:
              DWORD2(v65) = *(_DWORD *)(v15 + 32);
              if ( DWORD2(v65) > 0x40 )
              {
                v57 = v12;
                sub_C43780(&v65, v34);
                v12 = v57;
              }
              else
              {
                *(_QWORD *)&v65 = *(_QWORD *)(v15 + 24);
              }
              LOBYTE(v66) = 1;
            }
            v5 = (char *)v63[0];
            v13 = (*(v63[0] - 16) & 2) != 0;
            if ( a4 )
            {
              v36 = (*(v63[0] - 16) & 2) != 0
                  ? (const char *)*((_QWORD *)v63[0] - 4)
                  : &v63[0][-8 * (((unsigned __int8)*(v63[0] - 16) >> 2) & 0xF) - 16];
              v37 = *(_QWORD *)&v36[8 * v10 + 16];
              if ( !v37 || *(_BYTE *)v37 != 1 || **(_BYTE **)(v37 + 136) != 17 )
              {
                v38 = *a1;
                v64 = a2;
                if ( v38 )
                {
                  v69 = 1;
                  v67[0] = "Member size entries must be constants!";
                  v68 = 3;
                  sub_BF02E0(v38, (__int64)v67, &v64, v63);
                  v5 = (char *)v63[0];
                }
                v12 = a4;
                v13 = (*(v5 - 16) & 2) != 0;
              }
            }
            goto LABEL_26;
          }
        }
      }
      v23 = *a1;
      v12 = 1;
      if ( *a1 )
      {
        v69 = 1;
        v67[0] = "Offset entries must be constants!";
        v68 = 3;
        v43 = (_BYTE *)*v23;
        if ( *v23 )
        {
          v58 = (_BYTE *)*v23;
          sub_CA0E80(v67, *v23);
          v44 = (_BYTE *)*((_QWORD *)v58 + 4);
          if ( (unsigned __int64)v44 >= *((_QWORD *)v58 + 3) )
          {
            sub_CB5D20(v58, 10);
          }
          else
          {
            *((_QWORD *)v58 + 4) = v44 + 1;
            *v44 = 10;
          }
          v43 = (_BYTE *)*v23;
        }
        *((_BYTE *)v23 + 152) = 1;
        if ( v43 )
        {
          if ( *a2 <= 0x1Cu )
          {
            sub_A5C020(a2, (__int64)v43, 1, (__int64)(v23 + 2));
            v45 = *v23;
            v46 = *(_BYTE **)(*v23 + 32);
            if ( (unsigned __int64)v46 >= *(_QWORD *)(*v23 + 24) )
              goto LABEL_120;
LABEL_99:
            *(_QWORD *)(v45 + 32) = v46 + 1;
            *v46 = 10;
          }
          else
          {
            sub_A693B0((__int64)a2, v43, (__int64)(v23 + 2), 0);
            v45 = *v23;
            v46 = *(_BYTE **)(*v23 + 32);
            if ( (unsigned __int64)v46 < *(_QWORD *)(*v23 + 24) )
              goto LABEL_99;
LABEL_120:
            sub_CB5D20(v45, 10);
          }
          v5 = (char *)v63[0];
          if ( v63[0] )
          {
            v27 = v23[1];
            v28 = (__int64)(v23 + 2);
LABEL_38:
            sub_A62C00(v5, *v23, v28, v27);
            v29 = *v23;
            v30 = *(_BYTE **)(*v23 + 32);
            if ( (unsigned __int64)v30 < *(_QWORD *)(*v23 + 24) )
            {
              *(_QWORD *)(v29 + 32) = v30 + 1;
              *v30 = 10;
              goto LABEL_24;
            }
            sub_CB5D20(v29, 10);
            v5 = (char *)v63[0];
          }
LABEL_25:
          v12 = 1;
          v13 = (*(v5 - 16) & 2) != 0;
          goto LABEL_26;
        }
LABEL_24:
        v5 = (char *)v63[0];
        goto LABEL_25;
      }
    }
    else
    {
      v23 = *a1;
      v12 = 1;
      if ( *a1 )
      {
        v69 = 1;
        v67[0] = "Incorrect field entry in struct type node!";
        v68 = 3;
        v24 = *v23;
        if ( *v23 )
        {
          sub_CA0E80(v67, *v23);
          v25 = *(_BYTE **)(v24 + 32);
          if ( (unsigned __int64)v25 >= *(_QWORD *)(v24 + 24) )
          {
            sub_CB5D20(v24, 10);
          }
          else
          {
            *(_QWORD *)(v24 + 32) = v25 + 1;
            *v25 = 10;
          }
        }
        v26 = *v23 == 0;
        *((_BYTE *)v23 + 152) = 1;
        if ( !v26 )
        {
          sub_BDBD80((__int64)v23, a2);
          v5 = (char *)v63[0];
          if ( v63[0] )
          {
            v27 = v23[1];
            v28 = (__int64)(v23 + 2);
            goto LABEL_38;
          }
          goto LABEL_25;
        }
        goto LABEL_24;
      }
    }
LABEL_26:
    v10 += v11;
  }
  if ( v10 < ((*((_WORD *)v5 - 8) >> 6) & 0xFu) )
  {
    v20 = (__int64)&v5[-8 * (((unsigned __int8)*(v5 - 16) >> 2) & 0xF) - 16];
    v21 = v10 + 1;
    v22 = (_BYTE **)(v20 + 8LL * v10);
    goto LABEL_30;
  }
LABEL_82:
  result = v61 << 32;
  if ( v12 )
    result = 0xFFFFFFFF00000001LL;
  if ( (_BYTE)v66 )
  {
    LOBYTE(v66) = 0;
    if ( DWORD2(v65) > 0x40 )
    {
      if ( (_QWORD)v65 )
      {
        v62 = result;
        j_j___libc_free_0_0(v65);
        return v62;
      }
    }
  }
  return result;
}
