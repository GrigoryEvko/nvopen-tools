// Function: sub_1655620
// Address: 0x1655620
//
void __fastcall sub_1655620(__int64 a1, __int64 a2)
{
  __int64 v4; // rdx
  __int64 v5; // rcx
  _QWORD *v6; // rdi
  _BYTE **v7; // rbx
  _BYTE **v8; // r15
  _BYTE *v9; // rsi
  __int64 v10; // r12
  _BYTE *v11; // rax
  const char *v12; // rax
  __int64 v13; // rdx
  const char *v14; // rax
  __int64 v15; // rdx
  const char *v16; // rax
  __int64 v17; // rdx
  const char *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rbx
  __int64 v22; // rax
  __int64 *v23; // r14
  __int64 v24; // rax
  unsigned __int8 v25; // dl
  const char *v26; // rax
  __int64 v27; // rax
  __int64 v28; // r14
  _QWORD *v29; // rdi
  __int64 *v30; // rax
  __int64 *v31; // rax
  __int64 v32; // rbx
  __int64 v33; // rax
  __int64 v34; // r14
  _BYTE *v35; // rax
  __int64 v36; // rsi
  __int64 v37; // rdi
  _BYTE *v38; // rax
  const char *v39; // rax
  __int64 v40; // rax
  __int64 *v41; // rax
  __int64 *v42; // rax
  _BYTE *v43; // [rsp+0h] [rbp-70h] BYREF
  __int64 v44; // [rsp+8h] [rbp-68h]
  _BYTE v45[16]; // [rsp+10h] [rbp-60h] BYREF
  _QWORD v46[2]; // [rsp+20h] [rbp-50h] BYREF
  char v47; // [rsp+30h] [rbp-40h]
  char v48; // [rsp+31h] [rbp-3Fh]

  if ( !sub_15E4F60(a2) )
  {
    v6 = *(_QWORD **)(a2 - 24);
    if ( *(_QWORD *)(a2 + 24) != *v6 )
    {
      v48 = 1;
      v46[0] = "Global variable initializer type does not match global variable type!";
      v47 = 3;
      sub_164FF40((__int64 *)a1, (__int64)v46);
      if ( *(_QWORD *)a1 )
        sub_164FA80((__int64 *)a1, a2);
      return;
    }
    if ( (*(_BYTE *)(a2 + 32) & 0xF) == 0xA )
    {
      if ( !sub_1593BB0((__int64)v6, a2, v4, v5) )
      {
        v43 = (_BYTE *)a2;
        v39 = "'common' global must have a zero initializer!";
        v48 = 1;
        goto LABEL_65;
      }
      if ( (*(_BYTE *)(a2 + 80) & 1) != 0 )
      {
        v43 = (_BYTE *)a2;
        v39 = "'common' global may not be marked constant!";
        v48 = 1;
        goto LABEL_65;
      }
      if ( *(_QWORD *)(a2 + 48) )
      {
        v43 = (_BYTE *)a2;
        v39 = "'common' global may not be in a Comdat!";
        v48 = 1;
LABEL_65:
        v46[0] = v39;
        v47 = 3;
        sub_1655440((_BYTE *)a1, (__int64)v46, (__int64 *)&v43);
        return;
      }
    }
  }
  if ( (*(_BYTE *)(a2 + 23) & 0x20) != 0 )
  {
    v12 = sub_1649960(a2);
    if ( v13 == 17
      && !(*(_QWORD *)v12 ^ 0x6F6C672E6D766C6CLL | *((_QWORD *)v12 + 1) ^ 0x726F74635F6C6162LL)
      && v12[16] == 115
      || (v14 = sub_1649960(a2), v15 == 17)
      && !(*(_QWORD *)v14 ^ 0x6F6C672E6D766C6CLL | *((_QWORD *)v14 + 1) ^ 0x726F74645F6C6162LL)
      && v14[16] == 115 )
    {
      if ( !sub_15E4F60(a2) && (*(_BYTE *)(a2 + 32) & 0xF) != 6 )
        goto LABEL_74;
      v27 = *(_QWORD *)(a2 + 24);
      if ( *(_BYTE *)(v27 + 8) == 14 )
      {
        v28 = *(_QWORD *)(v27 + 24);
        v29 = *(_QWORD **)(a1 + 64);
        if ( *(_BYTE *)(v28 + 8) != 13 )
        {
          v41 = (__int64 *)sub_1643270(v29);
          v42 = (__int64 *)sub_16453E0(v41, 0);
          sub_1647190(v42, 0);
          goto LABEL_51;
        }
        v30 = (__int64 *)sub_1643270(v29);
        v31 = (__int64 *)sub_16453E0(v30, 0);
        v32 = sub_1647190(v31, 0);
        if ( (unsigned int)(*(_DWORD *)(v28 + 12) - 2) > 1
          || (v33 = sub_1643D80(v28, 0), !sub_1642F90(v33, 32))
          || v32 != sub_1643D80(v28, 1u) )
        {
LABEL_51:
          v34 = *(_QWORD *)a1;
          v48 = 1;
          v46[0] = "wrong type for intrinsic global variable";
          v47 = 3;
          if ( !v34 )
          {
            *(_BYTE *)(a1 + 72) = 1;
            return;
          }
          sub_16E2CE0(v46, v34);
          v35 = *(_BYTE **)(v34 + 24);
          if ( (unsigned __int64)v35 >= *(_QWORD *)(v34 + 16) )
          {
            sub_16E7DE0(v34, 10);
          }
          else
          {
            *(_QWORD *)(v34 + 24) = v35 + 1;
            *v35 = 10;
          }
          v36 = *(_QWORD *)a1;
          *(_BYTE *)(a1 + 72) = 1;
          if ( !v36 )
            return;
          if ( *(_BYTE *)(a2 + 16) <= 0x17u )
          {
            sub_1553920((__int64 *)a2, v36, 1, a1 + 16);
            v37 = *(_QWORD *)a1;
            v38 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
            if ( (unsigned __int64)v38 < *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
              goto LABEL_57;
          }
          else
          {
            sub_155BD40(a2, v36, a1 + 16, 0);
            v37 = *(_QWORD *)a1;
            v38 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
            if ( (unsigned __int64)v38 < *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
            {
LABEL_57:
              *(_QWORD *)(v37 + 24) = v38 + 1;
              *v38 = 10;
              return;
            }
          }
          sub_16E7DE0(v37, 10);
          return;
        }
        if ( *(_DWORD *)(v28 + 12) == 3 )
        {
          v40 = sub_1643D80(v28, 2u);
          if ( *(_BYTE *)(v40 + 8) != 15 || !sub_1642F90(*(_QWORD *)(v40 + 24), 8) )
            goto LABEL_73;
        }
      }
    }
    if ( (*(_BYTE *)(a2 + 23) & 0x20) == 0 )
      goto LABEL_5;
    v16 = sub_1649960(a2);
    if ( v17 != 9 || *(_QWORD *)v16 != 0x6573752E6D766C6CLL || v16[8] != 100 )
    {
      v18 = sub_1649960(a2);
      if ( v19 != 18
        || *(_QWORD *)v18 ^ 0x6D6F632E6D766C6CLL | *((_QWORD *)v18 + 1) ^ 0x73752E72656C6970LL
        || *((_WORD *)v18 + 8) != 25701 )
      {
        goto LABEL_5;
      }
    }
    if ( sub_15E4F60(a2) || (*(_BYTE *)(a2 + 32) & 0xF) == 6 )
    {
      v20 = *(_QWORD *)(a2 + 24);
      if ( *(_BYTE *)(v20 + 8) != 14 )
        goto LABEL_5;
      if ( *(_BYTE *)(*(_QWORD *)(v20 + 24) + 8LL) == 15 )
      {
        if ( !sub_15E4F60(a2) )
        {
          v21 = *(_QWORD *)(a2 - 24);
          if ( *(_BYTE *)(v21 + 16) == 6 )
          {
            v22 = 3LL * (*(_DWORD *)(v21 + 20) & 0xFFFFFFF);
            if ( (*(_BYTE *)(v21 + 23) & 0x40) != 0 )
            {
              v23 = *(__int64 **)(v21 - 8);
              v21 = (__int64)&v23[v22];
            }
            else
            {
              v23 = (__int64 *)(v21 - v22 * 8);
            }
            while ( 1 )
            {
              if ( (__int64 *)v21 == v23 )
                goto LABEL_5;
              v24 = sub_1649F00(*v23);
              v43 = (_BYTE *)v24;
              v25 = *(_BYTE *)(v24 + 16);
              if ( v25 != 3 && v25 > 1u )
                break;
              if ( (*(_BYTE *)(v24 + 23) & 0x20) == 0 )
              {
                v48 = 1;
                v26 = "members of llvm.used must be named";
LABEL_39:
                v46[0] = v26;
                v47 = 3;
                sub_1655530((_BYTE *)a1, (__int64)v46, (__int64 *)&v43);
                return;
              }
              v23 += 3;
            }
            v48 = 1;
            v26 = "invalid llvm.used member";
            goto LABEL_39;
          }
          v48 = 1;
          v46[0] = "wrong initalizer for intrinsic global variable";
          v47 = 3;
          sub_164FF40((__int64 *)a1, (__int64)v46);
          if ( *(_QWORD *)a1 )
            sub_164FA80((__int64 *)a1, v21);
          return;
        }
        goto LABEL_5;
      }
LABEL_73:
      v43 = (_BYTE *)a2;
      v39 = "wrong type for intrinsic global variable";
      v48 = 1;
      goto LABEL_65;
    }
LABEL_74:
    v43 = (_BYTE *)a2;
    v39 = "invalid linkage for intrinsic global variable";
    v48 = 1;
    goto LABEL_65;
  }
LABEL_5:
  v43 = v45;
  v44 = 0x100000000LL;
  sub_1626560(a2, 0, (__int64)&v43);
  v7 = (_BYTE **)v43;
  v8 = (_BYTE **)&v43[8 * (unsigned int)v44];
  if ( v43 == (_BYTE *)v8 )
  {
LABEL_8:
    if ( !sub_15E4F60(a2) )
    {
      sub_16501E0(a1, *(_QWORD *)(a2 - 24));
      sub_1651CA0((__int64 *)a1, a2);
      if ( v43 != v45 )
        _libc_free((unsigned __int64)v43);
      return;
    }
    sub_1651CA0((__int64 *)a1, a2);
  }
  else
  {
    while ( 1 )
    {
      v9 = *v7;
      if ( **v7 != 7 )
        break;
      ++v7;
      sub_1652F60(a1, (__int64)v9);
      if ( v8 == v7 )
        goto LABEL_8;
    }
    v10 = *(_QWORD *)a1;
    v48 = 1;
    v46[0] = "!dbg attachment of global variable must be a DIGlobalVariableExpression";
    v47 = 3;
    if ( v10 )
    {
      sub_16E2CE0(v46, v10);
      v11 = *(_BYTE **)(v10 + 24);
      if ( (unsigned __int64)v11 >= *(_QWORD *)(v10 + 16) )
      {
        sub_16E7DE0(v10, 10);
      }
      else
      {
        *(_QWORD *)(v10 + 24) = v11 + 1;
        *v11 = 10;
      }
    }
    *(_BYTE *)(a1 + 72) |= *(_BYTE *)(a1 + 74);
    *(_BYTE *)(a1 + 73) = 1;
  }
  if ( v43 != v45 )
    _libc_free((unsigned __int64)v43);
}
