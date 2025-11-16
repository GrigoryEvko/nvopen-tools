// Function: sub_12A5110
// Address: 0x12a5110
//
__int64 __fastcall sub_12A5110(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4)
{
  __int64 i; // r12
  __int64 v7; // rax
  __int64 v8; // r8
  __int64 v9; // rax
  const char **v10; // rax
  const char *v11; // r9
  size_t v12; // rax
  const char *v13; // r9
  void *v14; // r8
  size_t v15; // r11
  char *v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdi
  void *v21; // rdx
  char v22; // dl
  unsigned __int64 v23; // rax
  __int64 v24; // rax
  int v25; // eax
  __int64 v26; // rax
  __int64 v27; // r13
  __int64 v28; // r15
  __int64 v29; // rcx
  __int64 v31; // r15
  __int64 v32; // rax
  _QWORD *v33; // rdi
  size_t n; // [rsp+8h] [rbp-88h]
  __int64 v35; // [rsp+10h] [rbp-80h]
  const char *src; // [rsp+18h] [rbp-78h]
  void *srca; // [rsp+18h] [rbp-78h]
  void *srcb; // [rsp+18h] [rbp-78h]
  void *srcc; // [rsp+18h] [rbp-78h]
  __int64 v40; // [rsp+20h] [rbp-70h]
  __int64 v42; // [rsp+28h] [rbp-68h]
  size_t v43; // [rsp+38h] [rbp-58h] BYREF
  char *v44; // [rsp+40h] [rbp-50h] BYREF
  size_t v45; // [rsp+48h] [rbp-48h]
  _QWORD v46[8]; // [rsp+50h] [rbp-40h] BYREF

  for ( i = *(_QWORD *)(a2 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v7 = sub_1297B70((_QWORD **)(*(_QWORD *)(a1 + 32) + 8LL), i, (*(_BYTE *)(a2 + 199) & 8) != 0);
  v8 = a4;
  v40 = v7;
  v9 = *(_QWORD *)(a1 + 32);
  *(_QWORD *)(a1 + 120) = a4;
  if ( (*(_BYTE *)(v9 + 376) & 1) != 0 )
  {
    v10 = *(const char ***)(a2 + 256);
    if ( v10 )
    {
      v11 = *v10;
      if ( *v10 )
      {
        src = *v10;
        v44 = (char *)v46;
        v12 = strlen(v11);
        v13 = src;
        v14 = (void *)a4;
        v43 = v12;
        v15 = v12;
        if ( v12 > 0xF )
        {
          n = v12;
          v32 = sub_22409D0(&v44, &v43, 0);
          v13 = src;
          v14 = (void *)a4;
          v44 = (char *)v32;
          v33 = (_QWORD *)v32;
          v15 = n;
          v46[0] = v43;
        }
        else
        {
          if ( v12 == 1 )
          {
            LOBYTE(v46[0]) = *src;
            v16 = (char *)v46;
LABEL_9:
            v45 = v12;
            v16[v12] = 0;
            sub_15E5D20(v14, v44, v45);
            if ( v44 != (char *)v46 )
              j_j___libc_free_0(v44, v46[0] + 1LL);
            v8 = *(_QWORD *)(a1 + 120);
            goto LABEL_12;
          }
          if ( !v12 )
          {
            v16 = (char *)v46;
            goto LABEL_9;
          }
          v33 = v46;
        }
        srcc = v14;
        memcpy(v33, v13, v15);
        v12 = v43;
        v16 = v44;
        v14 = srcc;
        goto LABEL_9;
      }
    }
  }
LABEL_12:
  v42 = sub_12A4D50(a1, (__int64)"entry", v8, 0);
  v17 = sub_1643350(*(_QWORD *)(a1 + 40));
  v35 = sub_1599EF0(v17);
  v18 = sub_1643350(*(_QWORD *)(a1 + 40));
  LOWORD(v46[0]) = 257;
  srca = (void *)v18;
  v19 = sub_1648A60(56, 1);
  v20 = v19;
  if ( v19 )
  {
    v21 = srca;
    srcb = (void *)v19;
    sub_15FD650(v19, v35, v21, &v44, v42);
    v20 = (__int64)srcb;
  }
  *(_QWORD *)(a1 + 368) = v20;
  v44 = "allocapt";
  LOWORD(v46[0]) = 259;
  sub_164B780(v20, &v44);
  *(_QWORD *)(a1 + 128) = sub_12A4D50(a1, (__int64)"return", 0, 0);
  v22 = *(_BYTE *)(a3 + 140);
  if ( v22 == 12 )
  {
    v23 = a3;
    do
    {
      v23 = *(_QWORD *)(v23 + 160);
      v22 = *(_BYTE *)(v23 + 140);
    }
    while ( v22 == 12 );
  }
  if ( v22 != 1 )
  {
    v24 = *(_QWORD *)(v40 + 16);
    if ( *(_DWORD *)(v24 + 12) == 2 && sub_127B420(*(_QWORD *)(v24 + 24)) )
    {
      v31 = *(_QWORD *)(a1 + 120);
      if ( (*(_BYTE *)(v31 + 18) & 1) != 0 )
        sub_15E08E0(*(_QWORD *)(a1 + 120));
      *(_QWORD *)(a1 + 136) = *(_QWORD *)(v31 + 88);
      if ( *(char *)(a3 + 142) < 0 )
        goto LABEL_21;
    }
    else
    {
      v44 = "retval";
      LOWORD(v46[0]) = 259;
      *(_QWORD *)(a1 + 136) = sub_127FDC0((_QWORD *)a1, a3, (__int64)&v44);
      if ( *(char *)(a3 + 142) < 0 )
        goto LABEL_21;
    }
    if ( *(_BYTE *)(a3 + 140) == 12 )
    {
      v25 = sub_8D4AB0(a3);
      goto LABEL_22;
    }
LABEL_21:
    v25 = *(_DWORD *)(a3 + 136);
LABEL_22:
    *(_DWORD *)(a1 + 144) = v25;
    goto LABEL_23;
  }
  *(_QWORD *)(a1 + 136) = 0;
LABEL_23:
  *(_QWORD *)(a1 + 56) = v42;
  *(_QWORD *)(a1 + 64) = v42 + 40;
  v26 = *(_QWORD *)(a1 + 32);
  v27 = *(_QWORD *)(v26 + 384);
  if ( v27 )
    sub_129F050(*(_QWORD *)(v26 + 384));
  if ( dword_4D04658 )
  {
    if ( !dword_4D046B4 )
    {
LABEL_27:
      v28 = *(_QWORD *)(a1 + 120);
      goto LABEL_28;
    }
    goto LABEL_36;
  }
  sub_12A0360(v27, *(_DWORD *)(a2 + 64), *(_WORD *)(a2 + 68));
  if ( dword_4D046B4 )
  {
LABEL_36:
    sub_12A2620(v27, *(_QWORD *)(a1 + 120), a2);
    v28 = *(_QWORD *)(a1 + 120);
    goto LABEL_28;
  }
  v28 = *(_QWORD *)(a1 + 120);
  if ( !dword_4D04658 )
  {
    sub_12A0870(v27, v28, a2);
    goto LABEL_27;
  }
LABEL_28:
  v29 = *(_QWORD *)(a1 + 440);
  if ( v29 )
    v29 = *(_QWORD *)(v29 + 40);
  return sub_1297D00(
           (_QWORD *)a1,
           v40,
           v28,
           v29,
           **(_QWORD **)(i + 168),
           (_DWORD *)(a2 + 64),
           (*(_BYTE *)(a2 + 198) & 0x20) != 0);
}
