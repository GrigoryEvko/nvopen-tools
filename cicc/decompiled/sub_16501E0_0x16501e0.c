// Function: sub_16501E0
// Address: 0x16501e0
//
void __fastcall sub_16501E0(__int64 a1, __int64 a2)
{
  __int64 v4; // r12
  __int64 *v5; // rax
  char v6; // dl
  _QWORD *v7; // rdi
  unsigned int v8; // eax
  __int64 v9; // r14
  unsigned __int8 v10; // al
  unsigned __int16 v11; // ax
  int v12; // edx
  __int64 v13; // rbx
  __int64 v14; // r12
  _BYTE *v15; // rax
  __int64 v16; // rcx
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rcx
  void *v20; // rdx
  __int64 v21; // rax
  _WORD *v22; // rdx
  __int64 v23; // rdi
  void *v24; // rdx
  __int64 v25; // rax
  _WORD *v26; // rdx
  __int64 *v27; // rsi
  unsigned int v28; // edi
  __int64 *v29; // rcx
  const char *v30; // rdx
  __int64 v31; // rax
  bool v32; // zf
  __int64 v33; // rdx
  _DWORD *v34; // rdi
  _DWORD *v35; // rsi
  __int64 v36; // r8
  __int64 v37; // rbx
  _BYTE *v38; // rax
  unsigned __int16 v39; // ax
  __int64 v40; // rax
  _QWORD *v41; // r8
  __int64 v42; // r13
  _QWORD *v43; // r12
  _QWORD *v44; // r14
  char v45; // dl
  __int64 v46; // rax
  __int64 v47; // rbx
  _QWORD *v48; // rax
  _QWORD *v49; // rsi
  unsigned int v50; // edi
  _QWORD *v51; // rcx
  __int64 v52; // rax
  __int64 v53; // [rsp+8h] [rbp-108h]
  __int64 v54; // [rsp+10h] [rbp-100h]
  __int64 v55; // [rsp+10h] [rbp-100h]
  _QWORD v56[2]; // [rsp+20h] [rbp-F0h] BYREF
  int v57[4]; // [rsp+30h] [rbp-E0h] BYREF
  __int16 v58; // [rsp+40h] [rbp-D0h]
  _QWORD *v59; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v60; // [rsp+58h] [rbp-B8h]
  _QWORD v61[22]; // [rsp+60h] [rbp-B0h] BYREF

  v4 = a1 + 816;
  v5 = *(__int64 **)(a1 + 824);
  if ( *(__int64 **)(a1 + 832) != v5 )
  {
LABEL_2:
    sub_16CCBA0(v4, a2);
    if ( !v6 )
      return;
    goto LABEL_6;
  }
  v27 = &v5[*(unsigned int *)(a1 + 844)];
  v28 = *(_DWORD *)(a1 + 844);
  if ( v5 == v27 )
    goto LABEL_41;
  v29 = 0;
  do
  {
    if ( *v5 == a2 )
      return;
    if ( *v5 == -2 )
      v29 = v5;
    ++v5;
  }
  while ( v27 != v5 );
  if ( !v29 )
  {
LABEL_41:
    if ( v28 >= *(_DWORD *)(a1 + 840) )
      goto LABEL_2;
    *(_DWORD *)(a1 + 844) = v28 + 1;
    *v27 = a2;
    ++*(_QWORD *)(a1 + 816);
  }
  else
  {
    *v29 = a2;
    --*(_DWORD *)(a1 + 848);
    ++*(_QWORD *)(a1 + 816);
  }
LABEL_6:
  v7 = v61;
  v61[0] = a2;
  v59 = v61;
  v60 = 0x1000000001LL;
  v8 = 1;
  do
  {
    v9 = v7[v8 - 1];
    LODWORD(v60) = v8 - 1;
    v10 = *(_BYTE *)(v9 + 16);
    if ( v10 != 5 )
      goto LABEL_12;
    v11 = *(_WORD *)(v9 + 18);
    if ( v11 == 47 )
    {
      if ( !(unsigned __int8)sub_15FC090(
                               47,
                               *(_QWORD **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)),
                               *(_QWORD *)v9) )
      {
        *(_QWORD *)v57 = "Invalid bitcast";
        v58 = 259;
        sub_164FF40((__int64 *)a1, (__int64)v57);
        if ( *(_QWORD *)a1 )
          sub_164FA80((__int64 *)a1, v9);
LABEL_11:
        v10 = *(_BYTE *)(v9 + 16);
        goto LABEL_12;
      }
      v39 = *(_WORD *)(v9 + 18);
      v12 = v39;
      if ( v39 == 46 )
      {
LABEL_53:
        v31 = *(_QWORD *)v9;
        v30 = "inttoptr not supported for non-integral pointers";
        goto LABEL_44;
      }
    }
    else
    {
      v12 = v11;
      if ( v11 == 46 )
        goto LABEL_53;
    }
    if ( v12 != 45 )
      goto LABEL_11;
    v30 = "ptrtoint not supported for non-integral pointers";
    v31 = **(_QWORD **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF));
LABEL_44:
    v56[1] = 48;
    v32 = *(_BYTE *)(v31 + 8) == 16;
    v56[0] = v30;
    v33 = *(_QWORD *)(a1 + 56);
    if ( v32 )
      v31 = **(_QWORD **)(v31 + 16);
    v34 = *(_DWORD **)(v33 + 408);
    v35 = &v34[*(unsigned int *)(v33 + 416)];
    v57[0] = *(_DWORD *)(v31 + 8) >> 8;
    if ( v35 == sub_164E170(v34, (__int64)v35, v57) )
      goto LABEL_11;
    v37 = *(_QWORD *)a1;
    v58 = 261;
    *(_QWORD *)v57 = v56;
    if ( v37 )
    {
      sub_16E2CE0(v36, v37);
      v38 = *(_BYTE **)(v37 + 24);
      if ( (unsigned __int64)v38 >= *(_QWORD *)(v37 + 16) )
      {
        sub_16E7DE0(v37, 10);
      }
      else
      {
        *(_QWORD *)(v37 + 24) = v38 + 1;
        *v38 = 10;
      }
    }
    *(_BYTE *)(a1 + 72) = 1;
    v10 = *(_BYTE *)(v9 + 16);
LABEL_12:
    if ( v10 <= 3u )
    {
      v13 = *(_QWORD *)(v9 + 40);
      if ( *(_QWORD *)(a1 + 8) == v13 )
        goto LABEL_27;
      v14 = *(_QWORD *)a1;
      v54 = *(_QWORD *)(a1 + 8);
      *(_QWORD *)v57 = "Referencing global in another module!";
      v58 = 259;
      if ( v14 )
      {
        sub_16E2CE0(v57, v14);
        v15 = *(_BYTE **)(v14 + 24);
        v16 = v54;
        if ( (unsigned __int64)v15 >= *(_QWORD *)(v14 + 16) )
        {
          sub_16E7DE0(v14, 10);
          v17 = *(_QWORD *)a1;
          v16 = v54;
        }
        else
        {
          *(_QWORD *)(v14 + 24) = v15 + 1;
          *v15 = 10;
          v17 = *(_QWORD *)a1;
        }
        v55 = v16;
        *(_BYTE *)(a1 + 72) = 1;
        if ( v17 )
        {
          sub_164FA80((__int64 *)a1, a2);
          v18 = *(_QWORD *)a1;
          v19 = v55;
          v20 = *(void **)(*(_QWORD *)a1 + 24LL);
          if ( *(_QWORD *)(*(_QWORD *)a1 + 16LL) - (_QWORD)v20 <= 0xDu )
          {
            v52 = sub_16E7EE0(v18, "; ModuleID = '", 14, v55);
            v19 = v55;
            v18 = v52;
          }
          else
          {
            qmemcpy(v20, "; ModuleID = '", 14);
            *(_QWORD *)(v18 + 24) += 14LL;
          }
          v21 = sub_16E7EE0(v18, *(const char **)(v19 + 176), *(_QWORD *)(v19 + 184));
          v22 = *(_WORD **)(v21 + 24);
          if ( *(_QWORD *)(v21 + 16) - (_QWORD)v22 <= 1u )
          {
            sub_16E7EE0(v21, "'\n", 2);
          }
          else
          {
            *v22 = 2599;
            *(_QWORD *)(v21 + 24) += 2LL;
          }
          sub_164FA80((__int64 *)a1, v9);
          v23 = *(_QWORD *)a1;
          v24 = *(void **)(*(_QWORD *)a1 + 24LL);
          if ( *(_QWORD *)(*(_QWORD *)a1 + 16LL) - (_QWORD)v24 <= 0xDu )
          {
            v23 = sub_16E7EE0(v23, "; ModuleID = '", 14);
          }
          else
          {
            qmemcpy(v24, "; ModuleID = '", 14);
            *(_QWORD *)(v23 + 24) += 14LL;
          }
          v25 = sub_16E7EE0(v23, *(const char **)(v13 + 176), *(_QWORD *)(v13 + 184));
          v26 = *(_WORD **)(v25 + 24);
          if ( *(_QWORD *)(v25 + 16) - (_QWORD)v26 <= 1u )
          {
            sub_16E7EE0(v25, "'\n", 2);
          }
          else
          {
            *v26 = 2599;
            *(_QWORD *)(v25 + 24) += 2LL;
          }
        }
      }
      else
      {
        *(_BYTE *)(a1 + 72) = 1;
      }
      v7 = v59;
      if ( v59 != v61 )
        goto LABEL_39;
      return;
    }
    v40 = 3LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF);
    if ( (*(_BYTE *)(v9 + 23) & 0x40) != 0 )
    {
      v41 = *(_QWORD **)(v9 - 8);
      v9 = (__int64)&v41[v40];
    }
    else
    {
      v41 = (_QWORD *)(v9 - v40 * 8);
    }
    if ( (_QWORD *)v9 != v41 )
    {
      v53 = a2;
      v42 = v4;
      v43 = (_QWORD *)v9;
      v44 = v41;
      while ( 1 )
      {
        v47 = *v44;
        if ( *(_BYTE *)(*v44 + 16LL) <= 0x10u )
        {
          v48 = *(_QWORD **)(a1 + 824);
          if ( *(_QWORD **)(a1 + 832) == v48 )
          {
            v49 = &v48[*(unsigned int *)(a1 + 844)];
            v50 = *(_DWORD *)(a1 + 844);
            if ( v48 != v49 )
            {
              v51 = 0;
              while ( v47 != *v48 )
              {
                if ( *v48 == -2 )
                  v51 = v48;
                if ( v49 == ++v48 )
                {
                  if ( !v51 )
                    goto LABEL_78;
                  *v51 = v47;
                  --*(_DWORD *)(a1 + 848);
                  ++*(_QWORD *)(a1 + 816);
                  goto LABEL_59;
                }
              }
              goto LABEL_62;
            }
LABEL_78:
            if ( v50 < *(_DWORD *)(a1 + 840) )
            {
              *(_DWORD *)(a1 + 844) = v50 + 1;
              *v49 = v47;
              ++*(_QWORD *)(a1 + 816);
LABEL_59:
              v46 = (unsigned int)v60;
              if ( (unsigned int)v60 >= HIDWORD(v60) )
              {
                sub_16CD150(&v59, v61, 0, 8);
                v46 = (unsigned int)v60;
              }
              v59[v46] = v47;
              LODWORD(v60) = v60 + 1;
              goto LABEL_62;
            }
          }
          sub_16CCBA0(v42, *v44);
          if ( v45 )
            goto LABEL_59;
        }
LABEL_62:
        v44 += 3;
        if ( v43 == v44 )
        {
          v4 = v42;
          a2 = v53;
          break;
        }
      }
    }
LABEL_27:
    v8 = v60;
    v7 = v59;
  }
  while ( (_DWORD)v60 );
  if ( v59 != v61 )
LABEL_39:
    _libc_free((unsigned __int64)v7);
}
