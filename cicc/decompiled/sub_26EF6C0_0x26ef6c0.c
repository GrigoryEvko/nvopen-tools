// Function: sub_26EF6C0
// Address: 0x26ef6c0
//
void __fastcall sub_26EF6C0(_QWORD *a1, char a2)
{
  _BYTE *v3; // rax
  __int64 *v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 *v8; // rdx
  __int64 v9; // rcx
  _BYTE *v10; // rdi
  __int64 v11; // r8
  __int64 v12; // r9
  unsigned __int8 *v13; // r12
  unsigned __int8 *v14; // r14
  unsigned __int8 **v15; // rax
  unsigned __int8 **v16; // rdx
  unsigned __int8 *i; // rbx
  unsigned __int8 **v18; // rax
  unsigned __int8 **v19; // rdx
  __int64 **v20; // rax
  int v21; // ecx
  __int64 *v22; // rsi
  __int64 v23; // rax
  __int64 *v24; // rax
  __int64 v25; // rdx
  __int64 *v26; // r12
  __int64 v27; // rdx
  __int64 *v28; // r15
  __int64 v29; // rax
  unsigned __int8 *v30; // r14
  const char *v31; // rax
  unsigned __int64 v32; // rdx
  unsigned __int64 v33; // r14
  __int64 *v34; // rbx
  __int64 **v35; // r12
  __int64 v36; // rdx
  _QWORD *v37; // rax
  unsigned __int64 v38; // rdx
  const char *v39; // rax
  unsigned __int64 v40; // rdx
  const char *v41; // rax
  unsigned __int64 v42; // rdx
  __int64 v44; // [rsp+20h] [rbp-130h] BYREF
  unsigned __int8 **v45; // [rsp+28h] [rbp-128h]
  __int64 v46; // [rsp+30h] [rbp-120h]
  int v47; // [rsp+38h] [rbp-118h]
  char v48; // [rsp+3Ch] [rbp-114h]
  char v49; // [rsp+40h] [rbp-110h] BYREF
  const char *v50; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v51; // [rsp+88h] [rbp-C8h]
  __int64 v52; // [rsp+90h] [rbp-C0h]
  __int64 v53; // [rsp+98h] [rbp-B8h]
  __int64 v54; // [rsp+A0h] [rbp-B0h]
  __int64 v55; // [rsp+A8h] [rbp-A8h]
  __int64 v56; // [rsp+B0h] [rbp-A0h]
  __int64 v57; // [rsp+B8h] [rbp-98h]
  __int64 v58; // [rsp+C0h] [rbp-90h]
  __int64 v59; // [rsp+C8h] [rbp-88h]
  __int64 v60; // [rsp+D0h] [rbp-80h]
  __int64 v61; // [rsp+D8h] [rbp-78h]
  __int64 v62; // [rsp+E0h] [rbp-70h]
  __int64 v63; // [rsp+E8h] [rbp-68h]
  __int64 v64; // [rsp+F0h] [rbp-60h]
  __int64 v65; // [rsp+F8h] [rbp-58h]
  __int64 *v66; // [rsp+100h] [rbp-50h]
  unsigned __int64 v67; // [rsp+108h] [rbp-48h]
  __int64 v68; // [rsp+110h] [rbp-40h]
  char v69; // [rsp+118h] [rbp-38h]

  v44 = 0;
  v45 = (unsigned __int8 **)&v49;
  v46 = 8;
  v47 = 0;
  v48 = 1;
  v3 = sub_BA8CD0((__int64)a1, (__int64)"llvm.used", 9u, 0);
  if ( v3 )
    sub_26EF590((__int64)v3, (__int64)&v44, v4, v5, v6, v7);
  v10 = sub_BA8CD0((__int64)a1, (__int64)"llvm.compiler.used", 0x12u, 0);
  if ( v10 )
    sub_26EF590((__int64)v10, (__int64)&v44, v8, v9, v11, v12);
  v13 = (unsigned __int8 *)a1[2];
  v14 = (unsigned __int8 *)(a1 + 1);
  if ( a1 + 1 != (_QWORD *)v13 )
  {
    while ( 1 )
    {
      if ( !v13 )
        BUG();
      if ( (*(v13 - 24) & 0xFu) - 7 > 1 )
        goto LABEL_13;
      if ( v48 )
      {
        v15 = v45;
        v16 = &v45[HIDWORD(v46)];
        if ( v45 != v16 )
        {
          while ( v13 - 56 != *v15 )
          {
            if ( v16 == ++v15 )
              goto LABEL_57;
          }
          goto LABEL_13;
        }
LABEL_57:
        if ( a2 )
        {
          v39 = sub_BD5D20((__int64)(v13 - 56));
          if ( v40 > 7 && *(_QWORD *)v39 == 0x6762642E6D766C6CLL )
            goto LABEL_13;
        }
        LOWORD(v54) = 257;
        sub_BD6B50(v13 - 56, &v50);
        v13 = (unsigned __int8 *)*((_QWORD *)v13 + 1);
        if ( v14 == v13 )
          break;
      }
      else
      {
        if ( !sub_C8CA60((__int64)&v44, (__int64)(v13 - 56)) )
          goto LABEL_57;
LABEL_13:
        v13 = (unsigned __int8 *)*((_QWORD *)v13 + 1);
        if ( v14 == v13 )
          break;
      }
    }
  }
  for ( i = (unsigned __int8 *)a1[4]; a1 + 3 != (_QWORD *)i; i = (unsigned __int8 *)*((_QWORD *)i + 1) )
  {
    if ( !i )
      BUG();
    if ( (*(i - 24) & 0xFu) - 7 <= 1 )
    {
      if ( !v48 )
      {
        if ( sub_C8CA60((__int64)&v44, (__int64)(i - 56)) )
          goto LABEL_22;
LABEL_63:
        if ( !a2 || (v41 = sub_BD5D20((__int64)(i - 56)), v42 <= 7) || *(_QWORD *)v41 != 0x6762642E6D766C6CLL )
        {
          LOWORD(v54) = 257;
          sub_BD6B50(i - 56, &v50);
        }
        goto LABEL_22;
      }
      v18 = v45;
      v19 = &v45[HIDWORD(v46)];
      if ( v45 == v19 )
        goto LABEL_63;
      while ( i - 56 != *v18 )
      {
        if ( v19 == ++v18 )
          goto LABEL_63;
      }
    }
LABEL_22:
    v20 = (__int64 **)*((_QWORD *)i + 7);
    if ( v20 )
    {
      v21 = *((_DWORD *)v20 + 2);
      if ( v21 )
      {
        v22 = *v20;
        v23 = **v20;
        if ( v23 && v23 != -8 )
        {
          v24 = v22;
        }
        else
        {
          v24 = v22;
          do
          {
            do
            {
              v25 = v24[1];
              ++v24;
            }
            while ( v25 == -8 );
          }
          while ( !v25 );
        }
        v26 = &v22[v21];
        if ( v26 != v24 )
        {
          while ( 1 )
          {
            v27 = *v24;
            v28 = v24 + 1;
            v29 = v24[1];
            v30 = *(unsigned __int8 **)(v27 + 8);
            if ( v29 != -8 )
              goto LABEL_32;
            do
            {
              do
              {
                v29 = v28[1];
                ++v28;
              }
              while ( v29 == -8 );
LABEL_32:
              ;
            }
            while ( !v29 );
            if ( *v30 > 3u || (v30[32] & 0xFu) - 7 <= 1 )
            {
              if ( !a2 || (v31 = sub_BD5D20(*(_QWORD *)(v27 + 8)), v32 <= 7) || *(_QWORD *)v31 != 0x6762642E6D766C6CLL )
              {
                LOWORD(v54) = 257;
                sub_BD6B50(v30, &v50);
              }
            }
            if ( v26 == v28 )
              break;
            v24 = v28;
          }
        }
      }
    }
  }
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  v65 = 0;
  v66 = 0;
  v67 = 0;
  v68 = 0;
  v69 = 0;
  sub_BD22F0((__int64)&v50, a1, 0);
  v33 = v67;
  v34 = v66;
  if ( v66 != (__int64 *)v67 )
  {
    do
    {
      while ( 1 )
      {
        v35 = (__int64 **)*v34;
        if ( (*(_BYTE *)(*v34 + 9) & 4) == 0 )
        {
          sub_BCB490(*v34);
          if ( v36 )
          {
            if ( !a2 )
              break;
            v37 = (_QWORD *)sub_BCB490((__int64)v35);
            if ( v38 <= 7 || *v37 != 0x6762642E6D766C6CLL )
              break;
          }
        }
        if ( (__int64 *)v33 == ++v34 )
          goto LABEL_49;
      }
      ++v34;
      sub_BCB4B0(v35, byte_3F871B3, 0);
    }
    while ( (__int64 *)v33 != v34 );
LABEL_49:
    v33 = (unsigned __int64)v66;
  }
  if ( v33 )
    j_j___libc_free_0(v33);
  sub_C7D6A0(v63, 8LL * (unsigned int)v65, 8);
  sub_C7D6A0(v59, 8LL * (unsigned int)v61, 8);
  sub_C7D6A0(v55, 8LL * (unsigned int)v57, 8);
  sub_C7D6A0(v51, 8LL * (unsigned int)v53, 8);
  if ( !v48 )
    _libc_free((unsigned __int64)v45);
}
