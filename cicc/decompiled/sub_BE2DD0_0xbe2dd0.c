// Function: sub_BE2DD0
// Address: 0xbe2dd0
//
unsigned __int64 __fastcall sub_BE2DD0(__int64 *a1, __int64 a2)
{
  unsigned __int8 v4; // dl
  __int64 v5; // r14
  __int64 v6; // rsi
  unsigned __int8 *v7; // rcx
  unsigned __int64 v8; // rcx
  __int64 v9; // rdi
  const char *v10; // r15
  __int64 v11; // r14
  _BYTE *v12; // rax
  __int64 v13; // rdx
  unsigned __int64 result; // rax
  const char *v15; // rsi
  unsigned int v16; // ebx
  const char *v17; // r15
  unsigned int v18; // r15d
  const char *v19; // rbx
  unsigned __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r14
  _BYTE *v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rdi
  _BYTE *v26; // rax
  __int64 v27; // rcx
  __int64 v28; // rsi
  __int64 v29; // rdx
  const char *v30; // rdi
  _BYTE *v31; // rax
  __int64 v32; // r14
  _BYTE *v33; // rax
  char v34; // dl
  const char *v35; // rax
  __int64 v36; // r14
  _BYTE *v37; // rax
  __int64 v38; // rsi
  __int64 v39; // rdi
  _BYTE *v40; // rax
  __int64 v41; // rdi
  const char *v42; // rdx
  const char *v43; // rbx
  const char **v44; // rax
  __int64 v45; // rdx
  const char **v46; // rsi
  int v47; // eax
  const char *v48; // rax
  _BYTE *v49; // rax
  const char *v50; // r14
  __int64 v51; // rdx
  unsigned __int64 v52; // rsi
  __int64 v53; // r14
  _BYTE *v54; // rax
  const char *v55; // rax
  const char *v56; // rax
  const char *v57; // rax
  const char *v58; // [rsp+8h] [rbp-78h] BYREF
  const char *v59; // [rsp+10h] [rbp-70h] BYREF
  const char *v60; // [rsp+18h] [rbp-68h] BYREF
  _QWORD v61[4]; // [rsp+20h] [rbp-60h] BYREF
  char v62; // [rsp+40h] [rbp-40h]
  char v63; // [rsp+41h] [rbp-3Fh]

  if ( (unsigned __int16)sub_AF18C0(a2) != 46 )
  {
    v63 = 1;
    v61[0] = "invalid tag";
    v62 = 3;
    result = sub_BDD6D0(a1, (__int64)v61);
    if ( *a1 )
      return (unsigned __int64)sub_BD9900(a1, (const char *)a2);
    return result;
  }
  v4 = *(_BYTE *)(a2 - 16);
  v5 = a2 - 16;
  if ( (v4 & 2) != 0 )
    v6 = *(_QWORD *)(a2 - 32);
  else
    v6 = v5 - 8LL * ((v4 >> 2) & 0xF);
  v7 = *(unsigned __int8 **)(v6 + 8);
  if ( v7 )
  {
    v8 = *v7;
    if ( (unsigned __int8)v8 > 0x24u || (v9 = 0x16007FF000LL, !_bittest64(&v9, v8)) )
    {
      v31 = sub_A17150((_BYTE *)(a2 - 16));
      v32 = *a1;
      v10 = (const char *)*((_QWORD *)v31 + 1);
      v63 = 1;
      v61[0] = "invalid scope";
      v62 = 3;
      if ( v32 )
      {
        sub_CA0E80(v61, v32);
        v33 = *(_BYTE **)(v32 + 32);
        if ( (unsigned __int64)v33 >= *(_QWORD *)(v32 + 24) )
        {
          sub_CB5D20(v32, 10);
        }
        else
        {
          *(_QWORD *)(v32 + 32) = v33 + 1;
          *v33 = 10;
        }
        result = *a1;
        v34 = *((_BYTE *)a1 + 154);
        *((_BYTE *)a1 + 153) = 1;
        *((_BYTE *)a1 + 152) |= v34;
        if ( result )
        {
          result = (unsigned __int64)sub_BD9900(a1, (const char *)a2);
          if ( v10 )
            return (unsigned __int64)sub_BD9900(a1, v10);
        }
        return result;
      }
LABEL_42:
      result = *((unsigned __int8 *)a1 + 154);
      *((_BYTE *)a1 + 153) = 1;
      *((_BYTE *)a1 + 152) |= result;
      return result;
    }
  }
  if ( *(_BYTE *)a2 == 16 )
    goto LABEL_18;
  v10 = *(const char **)v6;
  if ( !*(_QWORD *)v6 )
  {
    v16 = *(_DWORD *)(a2 + 16);
    if ( v16 )
    {
      v63 = 1;
      v61[0] = "line specified with no file";
      v62 = 3;
      result = sub_BDD6D0(a1, (__int64)v61);
      if ( !*a1 )
        return result;
      sub_BD9900(a1, (const char *)a2);
      v41 = sub_CB59D0(*a1, v16);
      result = *(_QWORD *)(v41 + 32);
      if ( result < *(_QWORD *)(v41 + 24) )
      {
LABEL_54:
        *(_QWORD *)(v41 + 32) = result + 1;
        *(_BYTE *)result = 10;
        return result;
      }
      return sub_CB5D20(v41, 10);
    }
LABEL_18:
    if ( (v4 & 2) != 0 )
    {
      v17 = *(const char **)(*(_QWORD *)(a2 - 32) + 32LL);
      if ( !v17 || *v17 == 15 )
      {
        v18 = *(_DWORD *)(a2 - 24);
        goto LABEL_22;
      }
    }
    else
    {
      v17 = *(const char **)(v5 - 8LL * ((v4 >> 2) & 0xF) + 32);
      if ( !v17 || *v17 == 15 )
      {
        v18 = (*(_WORD *)(a2 - 16) >> 6) & 0xF;
LABEL_22:
        if ( v18 > 8 )
        {
          v19 = (const char *)*((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 8);
          if ( v19 )
          {
            v20 = *(unsigned __int8 *)v19;
            if ( (unsigned __int8)v20 > 0x24u || (v21 = 0x140000F000LL, !_bittest64(&v21, v20)) )
            {
              v22 = *a1;
              v63 = 1;
              v61[0] = "invalid containing type";
              v62 = 3;
              if ( !v22 )
                goto LABEL_42;
              sub_CA0E80(v61, v22);
              v23 = *(_BYTE **)(v22 + 32);
              if ( (unsigned __int64)v23 >= *(_QWORD *)(v22 + 24) )
              {
                sub_CB5D20(v22, 10);
              }
              else
              {
                *(_QWORD *)(v22 + 32) = v23 + 1;
                *v23 = 10;
              }
              v24 = *a1;
              result = *((unsigned __int8 *)a1 + 154);
              *((_BYTE *)a1 + 153) = 1;
              *((_BYTE *)a1 + 152) |= result;
              if ( !v24 )
                return result;
              sub_A62C00((const char *)a2, v24, (__int64)(a1 + 2), a1[1]);
              v25 = *a1;
              v26 = *(_BYTE **)(*a1 + 32);
              if ( (unsigned __int64)v26 >= *(_QWORD *)(*a1 + 24) )
              {
                sub_CB5D20(v25, 10);
              }
              else
              {
                *(_QWORD *)(v25 + 32) = v26 + 1;
                *v26 = 10;
              }
              v27 = a1[1];
              v28 = *a1;
              v29 = (__int64)(a1 + 2);
              v30 = v19;
LABEL_53:
              sub_A62C00(v30, v28, v29, v27);
              v41 = *a1;
              result = *(_QWORD *)(*a1 + 32);
              if ( result < *(_QWORD *)(*a1 + 24) )
                goto LABEL_54;
              return sub_CB5D20(v41, 10);
            }
          }
        }
        if ( v18 > 9 )
        {
          v42 = (const char *)*((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 9);
          if ( v42 )
            sub_BDB420((__int64)a1, (const char *)a2, v42);
        }
        v17 = (const char *)*((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 6);
        if ( !v17 || *v17 == 18 && (v17[36] & 8) == 0 )
        {
          v43 = (const char *)*((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 7);
          v59 = v43;
          if ( v43 )
          {
            if ( *v43 != 5 )
            {
              v60 = (const char *)a2;
              v56 = "invalid retained nodes list";
              v63 = 1;
              goto LABEL_119;
            }
            v44 = (const char **)sub_A17150((_BYTE *)v43 - 16);
            v46 = &v44[v45];
            if ( v44 != v46 )
            {
              while ( 1 )
              {
                v10 = *v44;
                if ( !*v44 )
                {
                  v63 = 1;
                  v61[0] = "invalid retained nodes, expected DILocalVariable, DILabel or DIImportedEntity";
                  v62 = 3;
                  result = sub_BDD6D0(a1, (__int64)v61);
                  if ( *a1 )
                  {
                    sub_BD9900(a1, (const char *)a2);
                    return (unsigned __int64)sub_BD9900(a1, v43);
                  }
                  return result;
                }
                if ( (unsigned __int8)(*v10 - 26) > 1u && *v10 != 29 )
                  break;
                if ( v46 == ++v44 )
                  goto LABEL_73;
              }
              v63 = 1;
              v61[0] = "invalid retained nodes, expected DILocalVariable, DILabel or DIImportedEntity";
              v62 = 3;
              result = sub_BDD6D0(a1, (__int64)v61);
              if ( !*a1 )
                return result;
              sub_BD9900(a1, (const char *)a2);
              v15 = v43;
              goto LABEL_15;
            }
          }
LABEL_73:
          v47 = *(_DWORD *)(a2 + 32);
          if ( (v47 & 0x6000) == 0x6000 || (v47 & 0xC00000) == 0xC00000 )
          {
            v53 = *a1;
            v63 = 1;
            v61[0] = "invalid reference flags";
            v62 = 3;
            if ( !v53 )
              goto LABEL_42;
            sub_CA0E80(v61, v53);
            v54 = *(_BYTE **)(v53 + 32);
            if ( (unsigned __int64)v54 >= *(_QWORD *)(v53 + 24) )
            {
              sub_CB5D20(v53, 10);
            }
            else
            {
              *(_QWORD *)(v53 + 32) = v54 + 1;
              *v54 = 10;
            }
            v28 = *a1;
            result = *((unsigned __int8 *)a1 + 154);
            *((_BYTE *)a1 + 153) = 1;
            *((_BYTE *)a1 + 152) |= result;
            if ( !v28 )
              return result;
            v27 = a1[1];
            v29 = (__int64)(a1 + 2);
            v30 = (const char *)a2;
            goto LABEL_53;
          }
          v48 = (const char *)*((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 5);
          v58 = v48;
          if ( (*(_BYTE *)(a2 + 36) & 8) != 0 )
          {
            if ( (*(_BYTE *)(a2 + 1) & 0x7F) == 1 )
            {
              if ( v48 )
              {
                if ( *v48 != 17 )
                {
                  v60 = (const char *)a2;
                  v63 = 1;
                  v61[0] = "invalid unit type";
                  v62 = 3;
                  return sub_BE2C90((__int64)a1, (__int64)v61, &v60, &v58);
                }
                v49 = (_BYTE *)*((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 1);
                if ( !v49
                  || *v49 != 14
                  || !*((_QWORD *)sub_A17150(v49 - 16) + 7)
                  || !(unsigned __int8)sub_B6F8F0(*(_QWORD *)a1[1])
                  || *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 6) )
                {
LABEL_84:
                  if ( (*(_BYTE *)(a2 - 16) & 2) != 0 )
                    result = *(unsigned int *)(a2 - 24);
                  else
                    result = (*(_WORD *)(a2 - 16) >> 6) & 0xF;
                  if ( (unsigned int)result <= 0xA
                    || (result = (unsigned __int64)sub_A17150((_BYTE *)(a2 - 16)),
                        (v50 = *(const char **)(result + 80)) == 0) )
                  {
LABEL_108:
                    if ( (*(_BYTE *)(a2 + 35) & 0x20) == 0 || (*(_BYTE *)(a2 + 36) & 8) != 0 )
                      return result;
                    v63 = 1;
                    v55 = "DIFlagAllCallsDescribed must be attached to a definition";
LABEL_111:
                    v61[0] = v55;
                    v62 = 3;
                    return sub_BDD6D0(a1, (__int64)v61);
                  }
                  v59 = *(const char **)(result + 80);
                  if ( *v50 == 5 )
                  {
                    result = (unsigned __int64)sub_A17150((_BYTE *)v50 - 16);
                    v52 = result + 8 * v51;
                    if ( v52 != result )
                    {
                      while ( 1 )
                      {
                        v10 = *(const char **)result;
                        if ( !*(_QWORD *)result )
                        {
                          v63 = 1;
                          v61[0] = "invalid thrown type";
                          v62 = 3;
                          result = sub_BDD6D0(a1, (__int64)v61);
                          if ( *a1 )
                          {
                            sub_BD9900(a1, (const char *)a2);
                            return (unsigned __int64)sub_BD9900(a1, v50);
                          }
                          return result;
                        }
                        if ( *v10 > 0x24u || ((1LL << *v10) & 0x140000F000LL) == 0 )
                          break;
                        result += 8LL;
                        if ( v52 == result )
                          goto LABEL_108;
                      }
                      v63 = 1;
                      v61[0] = "invalid thrown type";
                      v62 = 3;
                      result = sub_BDD6D0(a1, (__int64)v61);
                      if ( !*a1 )
                        return result;
                      sub_BD9900(a1, (const char *)a2);
                      v15 = v50;
                      goto LABEL_15;
                    }
                    goto LABEL_108;
                  }
                  v60 = (const char *)a2;
                  v56 = "invalid thrown types list";
                  v63 = 1;
LABEL_119:
                  v61[0] = v56;
                  v62 = 3;
                  return sub_BE2C90((__int64)a1, (__int64)v61, &v60, &v59);
                }
                v60 = (const char *)a2;
                v57 = "definition subprograms cannot be nested within DICompositeType when enabling ODR";
                v63 = 1;
              }
              else
              {
                v60 = (const char *)a2;
                v57 = "subprogram definitions must have a compile unit";
                v63 = 1;
              }
            }
            else
            {
              v60 = (const char *)a2;
              v57 = "subprogram definitions must be distinct";
              v63 = 1;
            }
          }
          else
          {
            if ( !v48 )
            {
              if ( *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 6) )
              {
                v63 = 1;
                v55 = "subprogram declaration must not have a declaration field";
                goto LABEL_111;
              }
              goto LABEL_84;
            }
            v60 = (const char *)a2;
            v57 = "subprogram declarations must not have a compile unit";
            v63 = 1;
          }
          v61[0] = v57;
          v62 = 3;
          return sub_BE2BA0(a1, (__int64)v61, &v60);
        }
        v63 = 1;
        v35 = "invalid subprogram declaration";
LABEL_46:
        v36 = *a1;
        v61[0] = v35;
        v62 = 3;
        if ( !v36 )
          goto LABEL_42;
        sub_CA0E80(v61, v36);
        v37 = *(_BYTE **)(v36 + 32);
        if ( (unsigned __int64)v37 >= *(_QWORD *)(v36 + 24) )
        {
          sub_CB5D20(v36, 10);
        }
        else
        {
          *(_QWORD *)(v36 + 32) = v37 + 1;
          *v37 = 10;
        }
        v38 = *a1;
        result = *((unsigned __int8 *)a1 + 154);
        *((_BYTE *)a1 + 153) = 1;
        *((_BYTE *)a1 + 152) |= result;
        if ( !v38 )
          return result;
        sub_A62C00((const char *)a2, v38, (__int64)(a1 + 2), a1[1]);
        v39 = *a1;
        v40 = *(_BYTE **)(*a1 + 32);
        if ( (unsigned __int64)v40 >= *(_QWORD *)(*a1 + 24) )
        {
          sub_CB5D20(v39, 10);
        }
        else
        {
          *(_QWORD *)(v39 + 32) = v40 + 1;
          *v40 = 10;
        }
        v27 = a1[1];
        v28 = *a1;
        v29 = (__int64)(a1 + 2);
        v30 = v17;
        goto LABEL_53;
      }
    }
    v63 = 1;
    v35 = "invalid subroutine type";
    goto LABEL_46;
  }
  if ( *v10 == 16 )
    goto LABEL_18;
  v11 = *a1;
  v63 = 1;
  v61[0] = "invalid file";
  v62 = 3;
  if ( !v11 )
    goto LABEL_42;
  sub_CA0E80(v61, v11);
  v12 = *(_BYTE **)(v11 + 32);
  if ( (unsigned __int64)v12 >= *(_QWORD *)(v11 + 24) )
  {
    sub_CB5D20(v11, 10);
  }
  else
  {
    *(_QWORD *)(v11 + 32) = v12 + 1;
    *v12 = 10;
  }
  v13 = *a1;
  result = *((unsigned __int8 *)a1 + 154);
  *((_BYTE *)a1 + 153) = 1;
  *((_BYTE *)a1 + 152) |= result;
  if ( v13 )
  {
    v15 = (const char *)a2;
LABEL_15:
    sub_BD9900(a1, v15);
    return (unsigned __int64)sub_BD9900(a1, v10);
  }
  return result;
}
