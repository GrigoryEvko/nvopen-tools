// Function: sub_1653200
// Address: 0x1653200
//
void __fastcall sub_1653200(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v6; // eax
  __int64 v7; // rax
  unsigned __int8 *v8; // r13
  int v9; // ecx
  __int64 v10; // r14
  _BYTE *v11; // rax
  __int64 v12; // r13
  _BYTE *v13; // rax
  __int64 v14; // rdx
  char v15; // al
  __int64 v16; // r13
  __int64 v17; // r10
  unsigned __int8 *i; // rax
  int v19; // edx
  unsigned __int8 *v20; // rdi
  unsigned __int8 *v21; // rcx
  int v22; // edx
  unsigned __int8 *v23; // r15
  unsigned __int8 *v24; // r10
  __int64 v25; // r14
  char v26; // al
  _BYTE *v27; // rax
  __int64 v28; // rdx
  char v29; // al
  __int64 v30; // rax
  __int64 v31; // r13
  unsigned int v32; // r15d
  unsigned __int64 v33; // rdx
  __int64 v34; // rax
  unsigned __int8 **v35; // rax
  unsigned __int8 *v36; // r14
  __int64 v37; // rdx
  void *v38; // rdi
  unsigned __int8 *v39; // [rsp+8h] [rbp-98h]
  unsigned __int8 *v40; // [rsp+10h] [rbp-90h]
  __int64 v41; // [rsp+18h] [rbp-88h]
  _QWORD v42[2]; // [rsp+20h] [rbp-80h] BYREF
  char *v43; // [rsp+30h] [rbp-70h] BYREF
  _QWORD *v44; // [rsp+38h] [rbp-68h]
  __int16 v45; // [rsp+40h] [rbp-60h]
  const char *v46; // [rsp+50h] [rbp-50h] BYREF
  const char *v47; // [rsp+58h] [rbp-48h]
  __int16 v48; // [rsp+60h] [rbp-40h]

  v6 = *(_DWORD *)(a4 + 20);
  v42[1] = a3;
  v42[0] = a2;
  v7 = v6 & 0xFFFFFFF;
  v8 = *(unsigned __int8 **)(*(_QWORD *)(a4 - 24 * v7) + 24LL);
  v9 = *v8;
  if ( (unsigned int)(v9 - 1) > 1 && ((unsigned __int8)(v9 - 4) > 0x1Eu || *((_DWORD *)v8 + 2)) )
  {
    v43 = "invalid llvm.dbg.";
    v44 = v42;
    v48 = 770;
    v25 = *(_QWORD *)a1;
    v46 = (const char *)&v43;
    v45 = 1283;
    v47 = " intrinsic address/value";
    if ( !v25 )
      goto LABEL_30;
    sub_16E2CE0(&v46, v25);
    v27 = *(_BYTE **)(v25 + 24);
    if ( (unsigned __int64)v27 >= *(_QWORD *)(v25 + 16) )
    {
      sub_16E7DE0(v25, 10);
    }
    else
    {
      *(_QWORD *)(v25 + 24) = v27 + 1;
      *v27 = 10;
    }
    v28 = *(_QWORD *)a1;
    v29 = *(_BYTE *)(a1 + 74);
    *(_BYTE *)(a1 + 73) = 1;
    *(_BYTE *)(a1 + 72) |= v29;
    if ( v28 )
      goto LABEL_34;
    return;
  }
  v10 = *(_QWORD *)(*(_QWORD *)(a4 + 24 * (1 - v7)) + 24LL);
  if ( *(_BYTE *)v10 != 25 )
  {
    v12 = *(_QWORD *)a1;
    v43 = "invalid llvm.dbg.";
    v44 = v42;
    v46 = (const char *)&v43;
    v45 = 1283;
    v47 = " intrinsic variable";
    v48 = 770;
    if ( !v12 )
    {
LABEL_30:
      v26 = *(_BYTE *)(a1 + 74);
      *(_BYTE *)(a1 + 73) = 1;
      *(_BYTE *)(a1 + 72) |= v26;
      return;
    }
    sub_16E2CE0(&v46, v12);
    v13 = *(_BYTE **)(v12 + 24);
    if ( (unsigned __int64)v13 >= *(_QWORD *)(v12 + 16) )
    {
      sub_16E7DE0(v12, 10);
    }
    else
    {
      *(_QWORD *)(v12 + 24) = v13 + 1;
      *v13 = 10;
    }
    v14 = *(_QWORD *)a1;
    v15 = *(_BYTE *)(a1 + 74);
    *(_BYTE *)(a1 + 73) = 1;
    *(_BYTE *)(a1 + 72) |= v15;
    if ( v14 )
    {
      sub_164FA80((__int64 *)a1, a4);
      sub_164ED40((__int64 *)a1, (unsigned __int8 *)v10);
    }
    return;
  }
  v8 = *(unsigned __int8 **)(*(_QWORD *)(a4 + 24 * (2 - v7)) + 24LL);
  if ( *v8 == 6 )
  {
    v11 = *(_BYTE **)(a4 + 48);
    if ( v11 && *v11 != 5 )
      return;
    v41 = 0;
    v16 = *(_QWORD *)(a4 + 40);
    if ( v16 )
      v41 = *(_QWORD *)(v16 + 56);
    v17 = sub_15C70A0(a4 + 48);
    if ( !v17 )
    {
      v43 = "llvm.dbg.";
      v44 = v42;
      v46 = (const char *)&v43;
      v48 = 770;
      v45 = 1283;
      v47 = " intrinsic requires a !dbg attachment";
      sub_16521E0((__int64 *)a1, (__int64)&v46);
      if ( *(_QWORD *)a1 )
      {
        sub_164FA80((__int64 *)a1, a4);
        sub_164FA80((__int64 *)a1, v16);
        sub_164FA80((__int64 *)a1, v41);
      }
      return;
    }
    for ( i = *(unsigned __int8 **)(v10 - 8LL * *(unsigned int *)(v10 + 8));
          i;
          i = *(unsigned __int8 **)&i[8 * (1LL - *((unsigned int *)i + 2))] )
    {
      v19 = *i;
      if ( (_BYTE)v19 == 17 )
        break;
      if ( (unsigned int)(v19 - 18) > 1 )
      {
        i = 0;
        break;
      }
    }
    v20 = *(unsigned __int8 **)(v17 - 8LL * *(unsigned int *)(v17 + 8));
    if ( v20 )
    {
      v21 = *(unsigned __int8 **)(v17 - 8LL * *(unsigned int *)(v17 + 8));
      while ( 1 )
      {
        v22 = *v21;
        if ( (_BYTE)v22 == 17 )
          break;
        if ( (unsigned int)(v22 - 18) <= 1 )
        {
          v21 = *(unsigned __int8 **)&v21[8 * (1LL - *((unsigned int *)v21 + 2))];
          if ( v21 )
            continue;
        }
        return;
      }
      v40 = (unsigned __int8 *)v17;
      if ( i )
      {
        if ( v21 != i )
        {
          v23 = sub_15B1000(v20);
          v39 = sub_15B1000(*(unsigned __int8 **)(v10 - 8LL * *(unsigned int *)(v10 + 8)));
          v45 = 1283;
          v43 = "mismatched subprogram between llvm.dbg.";
          v44 = v42;
          v47 = " variable and !dbg attachment";
          v46 = (const char *)&v43;
          v48 = 770;
          sub_16521E0((__int64 *)a1, (__int64)&v46);
          if ( *(_QWORD *)a1 )
          {
            sub_164FA80((__int64 *)a1, a4);
            sub_164FA80((__int64 *)a1, v16);
            sub_164FA80((__int64 *)a1, v41);
            sub_164ED40((__int64 *)a1, (unsigned __int8 *)v10);
            v24 = v40;
            if ( v39 )
            {
              sub_164ED40((__int64 *)a1, v39);
              v24 = v40;
            }
            sub_164ED40((__int64 *)a1, v24);
            if ( v23 )
              sub_164ED40((__int64 *)a1, v23);
          }
          return;
        }
        if ( *(_BYTE *)(a1 + 721) )
        {
          v30 = sub_15C70A0(a4 + 48);
          if ( *(_DWORD *)(v30 + 8) != 2 || !*(_QWORD *)(v30 - 8) )
          {
            v31 = *(_QWORD *)(*(_QWORD *)(a4 + 24 * (1LL - (*(_DWORD *)(a4 + 20) & 0xFFFFFFF))) + 24LL);
            if ( !v31 )
            {
              v46 = "dbg intrinsic without variable";
              v48 = 259;
              sub_16521E0((__int64 *)a1, (__int64)&v46);
              return;
            }
            v32 = *(unsigned __int16 *)(v31 + 32);
            if ( *(_WORD *)(v31 + 32) )
            {
              v33 = *(unsigned int *)(a1 + 1464);
              if ( v32 > (unsigned int)v33 )
              {
                if ( v33 > (unsigned __int16)v32 )
                {
                  *(_DWORD *)(a1 + 1464) = v32;
                  v34 = *(_QWORD *)(a1 + 1456);
                  goto LABEL_51;
                }
                if ( v33 < (unsigned __int16)v32 )
                {
                  if ( (unsigned __int16)v32 > (unsigned __int64)*(unsigned int *)(a1 + 1468) )
                    sub_16CD150(a1 + 1456, a1 + 1472, (unsigned __int16)v32, 8);
                  v34 = *(_QWORD *)(a1 + 1456);
                  v37 = *(unsigned int *)(a1 + 1464);
                  v38 = (void *)(v34 + 8 * v37);
                  if ( v38 != (void *)(v34 + 8LL * (unsigned __int16)v32) )
                  {
                    memset(v38, 0, 8 * ((unsigned __int16)v32 - v37));
                    v34 = *(_QWORD *)(a1 + 1456);
                  }
                  *(_DWORD *)(a1 + 1464) = v32;
                  goto LABEL_51;
                }
              }
              v34 = *(_QWORD *)(a1 + 1456);
LABEL_51:
              v35 = (unsigned __int8 **)(v34 + 8LL * (v32 - 1));
              v36 = *v35;
              *v35 = (unsigned __int8 *)v31;
              if ( (unsigned __int8 *)v31 != v36 )
              {
                if ( v36 )
                {
                  v46 = "conflicting debug info for argument";
                  v48 = 259;
                  sub_16521E0((__int64 *)a1, (__int64)&v46);
                  if ( *(_QWORD *)a1 )
                  {
                    sub_164FA80((__int64 *)a1, a4);
                    sub_164ED40((__int64 *)a1, v36);
                    sub_164ED40((__int64 *)a1, (unsigned __int8 *)v31);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v43 = "invalid llvm.dbg.";
    v44 = v42;
    v45 = 1283;
    v46 = (const char *)&v43;
    v47 = " intrinsic expression";
    v48 = 770;
    sub_16521E0((__int64 *)a1, (__int64)&v46);
    if ( *(_QWORD *)a1 )
    {
LABEL_34:
      sub_164FA80((__int64 *)a1, a4);
      sub_164ED40((__int64 *)a1, v8);
    }
  }
}
