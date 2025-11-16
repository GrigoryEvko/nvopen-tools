// Function: sub_946060
// Address: 0x946060
//
__int64 __fastcall sub_946060(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4)
{
  __int64 i; // r12
  __int64 v7; // rax
  __int64 v8; // r8
  __int64 v9; // rax
  const char **v10; // rax
  const char *v11; // r10
  size_t v12; // rax
  const char *v13; // r10
  void *v14; // r8
  size_t v15; // r9
  char *v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rax
  char *v19; // r10
  void *v20; // rdx
  char *v21; // rdx
  __int64 v22; // rdi
  __int64 v23; // rcx
  char v24; // dl
  unsigned __int64 v25; // rax
  __int64 v26; // rax
  int v27; // eax
  __int64 v28; // rax
  __int64 v29; // r13
  __int64 v30; // r15
  __int64 v31; // rcx
  __int64 v33; // r15
  __int64 v34; // rax
  _QWORD *v35; // rdi
  size_t n; // [rsp+0h] [rbp-A0h]
  __int64 v37; // [rsp+10h] [rbp-90h]
  char *v38; // [rsp+10h] [rbp-90h]
  const char *src; // [rsp+18h] [rbp-88h]
  void *srca; // [rsp+18h] [rbp-88h]
  char *srcb; // [rsp+18h] [rbp-88h]
  void *srcc; // [rsp+18h] [rbp-88h]
  __int64 v43; // [rsp+20h] [rbp-80h]
  __int64 v45; // [rsp+28h] [rbp-78h]
  _QWORD v46[2]; // [rsp+30h] [rbp-70h] BYREF
  char *v47; // [rsp+40h] [rbp-60h] BYREF
  size_t v48; // [rsp+48h] [rbp-58h]
  _QWORD v49[2]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v50; // [rsp+60h] [rbp-40h]

  for ( i = *(_QWORD *)(a2 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v7 = sub_9380F0((_QWORD **)(*(_QWORD *)(a1 + 32) + 8LL), i, (*(_BYTE *)(a2 + 199) & 8) != 0);
  v8 = a4;
  v43 = v7;
  v9 = *(_QWORD *)(a1 + 32);
  *(_QWORD *)(a1 + 192) = a4;
  if ( (*(_BYTE *)(v9 + 360) & 1) != 0 )
  {
    v10 = *(const char ***)(a2 + 256);
    if ( v10 )
    {
      v11 = *v10;
      if ( *v10 )
      {
        src = *v10;
        v47 = (char *)v49;
        v12 = strlen(v11);
        v13 = src;
        v14 = (void *)a4;
        v46[0] = v12;
        v15 = v12;
        if ( v12 > 0xF )
        {
          n = v12;
          v34 = sub_22409D0(&v47, v46, 0);
          v13 = src;
          v14 = (void *)a4;
          v47 = (char *)v34;
          v35 = (_QWORD *)v34;
          v15 = n;
          v49[0] = v46[0];
        }
        else
        {
          if ( v12 == 1 )
          {
            LOBYTE(v49[0]) = *src;
            v16 = (char *)v49;
LABEL_9:
            v48 = v12;
            v16[v12] = 0;
            sub_B31A00(v14, v47, v48);
            if ( v47 != (char *)v49 )
              j_j___libc_free_0(v47, v49[0] + 1LL);
            v8 = *(_QWORD *)(a1 + 192);
            goto LABEL_12;
          }
          if ( !v12 )
          {
            v16 = (char *)v49;
            goto LABEL_9;
          }
          v35 = v49;
        }
        srcc = v14;
        memcpy(v35, v13, v15);
        v12 = v46[0];
        v16 = v47;
        v14 = srcc;
        goto LABEL_9;
      }
    }
  }
LABEL_12:
  v45 = sub_945CA0(a1, (__int64)"entry", v8, 0);
  v17 = sub_BCB2D0(*(_QWORD *)(a1 + 40));
  v37 = sub_ACA8A0(v17);
  srca = (void *)sub_BCB2D0(*(_QWORD *)(a1 + 40));
  v50 = 257;
  sub_B43C20(v46, v45);
  v18 = sub_BD2C40(72, unk_3F10A14);
  v19 = (char *)v18;
  if ( v18 )
  {
    v20 = srca;
    srcb = (char *)v18;
    sub_B51BF0(v18, v37, v20, &v47, v46[0], v46[1]);
    v19 = srcb;
  }
  v21 = *(char **)(a1 + 456);
  v22 = a1 + 440;
  if ( v19 != v21 )
  {
    if ( v21 + 4096 != 0 && v21 != 0 && v21 != (char *)-8192LL )
    {
      v38 = v19;
      sub_BD60C0(v22);
      v19 = v38;
      v22 = a1 + 440;
    }
    *(_QWORD *)(a1 + 456) = v19;
    if ( v19 + 4096 != 0 && v19 != 0 && v19 != (char *)-8192LL )
    {
      sub_BD73F0(v22);
      v19 = *(char **)(a1 + 456);
    }
  }
  v47 = "allocapt";
  v50 = 259;
  sub_BD6B50(v19, &v47);
  *(_QWORD *)(a1 + 200) = sub_945CA0(a1, (__int64)"return", 0, 0);
  v24 = *(_BYTE *)(a3 + 140);
  if ( v24 == 12 )
  {
    v25 = a3;
    do
    {
      v25 = *(_QWORD *)(v25 + 160);
      v24 = *(_BYTE *)(v25 + 140);
    }
    while ( v24 == 12 );
  }
  if ( v24 != 1 )
  {
    v26 = *(_QWORD *)(v43 + 16);
    if ( *(_DWORD *)(v26 + 12) == 2 && sub_91B770(*(_QWORD *)(v26 + 24)) )
    {
      v33 = *(_QWORD *)(a1 + 192);
      if ( (*(_BYTE *)(v33 + 2) & 1) != 0 )
        sub_B2C6D0(*(_QWORD *)(a1 + 192));
      *(_QWORD *)(a1 + 208) = *(_QWORD *)(v33 + 96);
      if ( *(char *)(a3 + 142) < 0 )
        goto LABEL_28;
    }
    else
    {
      v47 = "retval";
      v50 = 259;
      *(_QWORD *)(a1 + 208) = sub_921CE0(a1, a3, (__int64)&v47, v23);
      if ( *(char *)(a3 + 142) < 0 )
        goto LABEL_28;
    }
    if ( *(_BYTE *)(a3 + 140) == 12 )
    {
      v27 = sub_8D4AB0(a3);
      goto LABEL_29;
    }
LABEL_28:
    v27 = *(_DWORD *)(a3 + 136);
LABEL_29:
    *(_DWORD *)(a1 + 216) = v27;
    goto LABEL_30;
  }
  *(_QWORD *)(a1 + 208) = 0;
LABEL_30:
  *(_WORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 96) = v45;
  *(_QWORD *)(a1 + 104) = v45 + 48;
  v28 = *(_QWORD *)(a1 + 32);
  v29 = *(_QWORD *)(v28 + 368);
  if ( v29 )
    sub_93FB10(*(_QWORD *)(v28 + 368));
  if ( dword_4D04658 )
  {
    if ( !dword_4D046B4 )
    {
LABEL_34:
      v30 = *(_QWORD *)(a1 + 192);
      goto LABEL_35;
    }
    goto LABEL_43;
  }
  sub_941230(v29, *(_DWORD *)(a2 + 64), *(_WORD *)(a2 + 68));
  if ( dword_4D046B4 )
  {
LABEL_43:
    sub_9435B0(v29, *(_QWORD *)(a1 + 192), a2);
    v30 = *(_QWORD *)(a1 + 192);
    goto LABEL_35;
  }
  v30 = *(_QWORD *)(a1 + 192);
  if ( !dword_4D04658 )
  {
    sub_9417D0(v29, v30, a2);
    goto LABEL_34;
  }
LABEL_35:
  v31 = *(_QWORD *)(a1 + 528);
  if ( v31 )
    v31 = *(_QWORD *)(v31 + 40);
  return sub_938240(a1, v43, v30, v31, **(_QWORD **)(i + 168), (_DWORD *)(a2 + 64), (*(_BYTE *)(a2 + 198) & 0x20) != 0);
}
