// Function: sub_1650C20
// Address: 0x1650c20
//
void __fastcall sub_1650C20(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  _QWORD *v6; // rax
  char v7; // dl
  __int64 v8; // rbx
  _QWORD *v9; // r12
  unsigned __int8 v10; // al
  _QWORD *v11; // r15
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // r14
  __int64 v15; // r11
  _BYTE *v16; // rax
  bool v17; // zf
  __int64 v18; // rdi
  __int64 v19; // rcx
  void *v20; // rdx
  __int64 *v21; // r14
  __int64 v22; // r15
  __int64 v23; // rax
  __int64 v24; // rdi
  void *v25; // rdx
  __int64 v26; // rax
  _WORD *v27; // rdx
  __int64 v28; // rdi
  __int64 v29; // r9
  __int64 v30; // r14
  _BYTE *v31; // rax
  __int64 v32; // rdi
  __int64 v33; // rcx
  void *v34; // rdx
  __int64 v35; // rax
  _WORD *v36; // rdx
  __int64 v37; // rdx
  _QWORD *v38; // rsi
  _QWORD *v39; // rcx
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rax
  _WORD *v43; // rdx
  void *v44; // rdx
  __int64 v45; // rax
  _WORD *v46; // rdx
  __int64 v47; // [rsp+0h] [rbp-80h]
  __int64 v48; // [rsp+8h] [rbp-78h]
  __int64 v49; // [rsp+10h] [rbp-70h]
  __int64 v50; // [rsp+10h] [rbp-70h]
  __int64 v51; // [rsp+10h] [rbp-70h]
  __int64 v52; // [rsp+10h] [rbp-70h]
  __int64 v53; // [rsp+18h] [rbp-68h]
  __int64 v54; // [rsp+18h] [rbp-68h]
  __int64 v55; // [rsp+18h] [rbp-68h]
  __int64 v56; // [rsp+20h] [rbp-60h]
  _QWORD v58[2]; // [rsp+30h] [rbp-50h] BYREF
  char v59; // [rsp+40h] [rbp-40h]
  char v60; // [rsp+41h] [rbp-3Fh]

  v6 = *(_QWORD **)(a2 + 8);
  if ( *(_QWORD **)(a2 + 16) != v6 )
  {
LABEL_2:
    sub_16CCBA0(a2, a1);
    if ( !v7 )
      return;
    goto LABEL_3;
  }
  v37 = *(unsigned int *)(a2 + 28);
  v38 = &v6[v37];
  if ( v6 == v38 )
    goto LABEL_56;
  v39 = 0;
  do
  {
    if ( a1 == *v6 )
      return;
    if ( *v6 == -2 )
      v39 = v6;
    ++v6;
  }
  while ( v38 != v6 );
  if ( !v39 )
  {
LABEL_56:
    if ( (unsigned int)v37 >= *(_DWORD *)(a2 + 24) )
      goto LABEL_2;
    *(_DWORD *)(a2 + 28) = v37 + 1;
    *v38 = a1;
    ++*(_QWORD *)a2;
  }
  else
  {
    *v39 = a1;
    --*(_DWORD *)(a2 + 32);
    ++*(_QWORD *)a2;
  }
LABEL_3:
  v8 = *(_QWORD *)(a1 + 8);
  if ( v8 )
  {
    v56 = a2;
    do
    {
      while ( 1 )
      {
        v9 = sub_1648700(v8);
        v10 = *((_BYTE *)v9 + 16);
        if ( v10 > 0x17u )
          break;
        if ( v10 )
        {
          sub_1650C20(v9, v56, a3, a4);
LABEL_16:
          v8 = *(_QWORD *)(v8 + 8);
          if ( !v8 )
            return;
        }
        else
        {
          v21 = (__int64 *)a4[1];
          v22 = v21[1];
          v53 = v9[5];
          if ( v53 == v22 )
            goto LABEL_16;
          v23 = *a4;
          v60 = 1;
          v59 = 3;
          v51 = v23;
          v58[0] = "Global is used by function in a different module";
          sub_164FF40(v21, (__int64)v58);
          if ( !*v21 )
            goto LABEL_16;
          sub_164FA80(v21, v51);
          v24 = *v21;
          v25 = *(void **)(*v21 + 24);
          if ( *(_QWORD *)(*v21 + 16) - (_QWORD)v25 <= 0xDu )
          {
            v24 = sub_16E7EE0(v24, "; ModuleID = '", 14);
          }
          else
          {
            qmemcpy(v25, "; ModuleID = '", 14);
            *(_QWORD *)(v24 + 24) += 14LL;
          }
          v26 = sub_16E7EE0(v24, *(const char **)(v22 + 176), *(_QWORD *)(v22 + 184));
          v27 = *(_WORD **)(v26 + 24);
          if ( *(_QWORD *)(v26 + 16) - (_QWORD)v27 <= 1u )
          {
            sub_16E7EE0(v26, "'\n", 2);
          }
          else
          {
            *v27 = 2599;
            *(_QWORD *)(v26 + 24) += 2LL;
          }
          sub_164FA80(v21, (__int64)v9);
          v28 = *v21;
LABEL_52:
          v44 = *(void **)(v28 + 24);
          if ( *(_QWORD *)(v28 + 16) - (_QWORD)v44 <= 0xDu )
          {
            v28 = sub_16E7EE0(v28, "; ModuleID = '", 14);
          }
          else
          {
            qmemcpy(v44, "; ModuleID = '", 14);
            *(_QWORD *)(v28 + 24) += 14LL;
          }
          v45 = sub_16E7EE0(v28, *(const char **)(v53 + 176), *(_QWORD *)(v53 + 184));
          v46 = *(_WORD **)(v45 + 24);
          if ( *(_QWORD *)(v45 + 16) - (_QWORD)v46 > 1u )
          {
            *v46 = 2599;
            *(_QWORD *)(v45 + 24) += 2LL;
            goto LABEL_16;
          }
          sub_16E7EE0(v45, "'\n", 2);
          v8 = *(_QWORD *)(v8 + 8);
          if ( !v8 )
            return;
        }
      }
      v11 = (_QWORD *)a4[1];
      v12 = v9[5];
      v13 = v11[1];
      if ( v12 )
      {
        v14 = *(_QWORD *)(v12 + 56);
        if ( v14 )
        {
          v53 = *(_QWORD *)(v14 + 40);
          if ( v53 == v13 )
            goto LABEL_16;
          v60 = 1;
          v15 = *a4;
          v58[0] = "Global is referenced in a different module!";
          v59 = 3;
          if ( *v11 )
          {
            v47 = v13;
            v48 = v15;
            v49 = *v11;
            sub_16E2CE0(v58, *v11);
            v15 = v48;
            v13 = v47;
            v16 = *(_BYTE **)(v49 + 24);
            if ( (unsigned __int64)v16 >= *(_QWORD *)(v49 + 16) )
            {
              sub_16E7DE0(v49, 10);
              v15 = v48;
              v13 = v47;
            }
            else
            {
              *(_QWORD *)(v49 + 24) = v16 + 1;
              *v16 = 10;
            }
          }
          v17 = *v11 == 0;
          v50 = v13;
          *((_BYTE *)v11 + 72) = 1;
          if ( v17 )
            goto LABEL_16;
          sub_164FA80(v11, v15);
          v18 = *v11;
          v19 = v50;
          v20 = *(void **)(*v11 + 24LL);
          if ( *(_QWORD *)(*v11 + 16LL) - (_QWORD)v20 <= 0xDu )
          {
            v41 = sub_16E7EE0(v18, "; ModuleID = '", 14, v50);
            v19 = v50;
            v18 = v41;
          }
          else
          {
            qmemcpy(v20, "; ModuleID = '", 14);
            *(_QWORD *)(v18 + 24) += 14LL;
          }
          v42 = sub_16E7EE0(v18, *(const char **)(v19 + 176), *(_QWORD *)(v19 + 184));
          v43 = *(_WORD **)(v42 + 24);
          if ( *(_QWORD *)(v42 + 16) - (_QWORD)v43 <= 1u )
          {
            sub_16E7EE0(v42, "'\n", 2);
          }
          else
          {
            *v43 = 2599;
            *(_QWORD *)(v42 + 24) += 2LL;
          }
          sub_164FA80(v11, (__int64)v9);
          sub_164FA80(v11, v14);
          v28 = *v11;
          goto LABEL_52;
        }
      }
      v60 = 1;
      v29 = *a4;
      v58[0] = "Global is referenced by parentless instruction!";
      v59 = 3;
      v30 = *v11;
      if ( *v11 )
      {
        v52 = v13;
        v54 = v29;
        sub_16E2CE0(v58, *v11);
        v31 = *(_BYTE **)(v30 + 24);
        v29 = v54;
        v13 = v52;
        if ( (unsigned __int64)v31 >= *(_QWORD *)(v30 + 16) )
        {
          sub_16E7DE0(v30, 10);
          v29 = v54;
          v13 = v52;
        }
        else
        {
          *(_QWORD *)(v30 + 24) = v31 + 1;
          *v31 = 10;
        }
      }
      v17 = *v11 == 0;
      v55 = v13;
      *((_BYTE *)v11 + 72) = 1;
      if ( v17 )
        goto LABEL_16;
      sub_164FA80(v11, v29);
      v32 = *v11;
      v33 = v55;
      v34 = *(void **)(*v11 + 24LL);
      if ( *(_QWORD *)(*v11 + 16LL) - (_QWORD)v34 <= 0xDu )
      {
        v40 = sub_16E7EE0(v32, "; ModuleID = '", 14, v55);
        v33 = v55;
        v32 = v40;
      }
      else
      {
        qmemcpy(v34, "; ModuleID = '", 14);
        *(_QWORD *)(v32 + 24) += 14LL;
      }
      v35 = sub_16E7EE0(v32, *(const char **)(v33 + 176), *(_QWORD *)(v33 + 184));
      v36 = *(_WORD **)(v35 + 24);
      if ( *(_QWORD *)(v35 + 16) - (_QWORD)v36 <= 1u )
      {
        sub_16E7EE0(v35, "'\n", 2);
      }
      else
      {
        *v36 = 2599;
        *(_QWORD *)(v35 + 24) += 2LL;
      }
      sub_164FA80(v11, (__int64)v9);
      v8 = *(_QWORD *)(v8 + 8);
    }
    while ( v8 );
  }
}
