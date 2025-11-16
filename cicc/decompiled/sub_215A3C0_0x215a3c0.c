// Function: sub_215A3C0
// Address: 0x215a3c0
//
void __fastcall sub_215A3C0(__int64 a1)
{
  __int64 v2; // rax
  _BYTE *v3; // r15
  __int64 v4; // rax
  __int64 i; // r14
  _QWORD *v6; // rax
  __int64 v7; // r15
  __int64 v8; // rcx
  __int64 v9; // r14
  unsigned __int64 v10; // r15
  __int64 v11; // rdi
  __int64 v12; // rdx
  __int64 *v13; // rax
  int v14; // eax
  _QWORD *v15; // r14
  __int64 v16; // rsi
  _QWORD *v17; // r15
  unsigned int v18; // edx
  __int64 v19; // rdi
  __int64 v20; // r14
  __int64 v21; // rax
  __int64 v22; // rax
  char v23; // al
  __int64 v24; // rcx
  char v25; // al
  __int64 v26; // rax
  __int64 v27; // rax
  unsigned int v28; // r15d
  void (__fastcall *v29)(__int64, const char *, __int64, _QWORD, __int64); // rbx
  __int64 v30; // rdx
  __int64 v31; // rax
  const char *v32; // rsi
  __int64 v33; // rdx
  const char *v34; // rax
  __int64 v35; // rdi
  int v36; // esi
  __int64 v37; // rdx
  unsigned int v38; // eax
  _QWORD *v39; // rdi
  unsigned __int64 v40; // rdx
  unsigned __int64 v41; // rax
  _QWORD *v42; // rax
  __int64 v43; // rdx
  _QWORD *j; // rdx
  _QWORD *v45; // rax
  int v46; // [rsp+Ch] [rbp-144h]
  int v47; // [rsp+Ch] [rbp-144h]
  __int64 v48; // [rsp+18h] [rbp-138h]
  __int64 v49; // [rsp+18h] [rbp-138h]
  __int64 v50; // [rsp+18h] [rbp-138h]
  __int64 *v51; // [rsp+20h] [rbp-130h] BYREF
  __int64 v52; // [rsp+28h] [rbp-128h]
  __int16 v53; // [rsp+30h] [rbp-120h]
  __int64 v54[2]; // [rsp+40h] [rbp-110h] BYREF
  __int64 v55; // [rsp+50h] [rbp-100h] BYREF
  _QWORD v56[4]; // [rsp+60h] [rbp-F0h] BYREF
  int v57; // [rsp+80h] [rbp-D0h]
  unsigned __int64 *v58; // [rsp+88h] [rbp-C8h]
  unsigned __int64 v59[2]; // [rsp+90h] [rbp-C0h] BYREF
  _BYTE v60[176]; // [rsp+A0h] [rbp-B0h] BYREF

  v59[1] = 0x8000000000LL;
  v59[0] = (unsigned __int64)v60;
  v56[0] = &unk_49EFC48;
  v57 = 1;
  memset(&v56[1], 0, 24);
  v58 = v59;
  sub_16E7A40((__int64)v56, 0, 0, 0);
  if ( !*(_BYTE *)(a1 + 784) )
  {
    sub_215A100(a1, *(_QWORD *)(**(_QWORD **)(a1 + 264) + 40LL));
    *(_BYTE *)(a1 + 784) = 1;
  }
  v2 = *(_QWORD *)(a1 + 264);
  *(_QWORD *)(a1 + 800) = *(_QWORD *)(v2 + 40);
  v3 = *(_BYTE **)v2;
  *(_QWORD *)(a1 + 744) = *(_QWORD *)v2;
  if ( LOBYTE(qword_5056180[20]) )
  {
    v27 = sub_1626D20((__int64)v3);
    if ( v27 )
    {
      v28 = *(_DWORD *)(v27 + 24);
      v29 = *(void (__fastcall **)(__int64, const char *, __int64, _QWORD, __int64))(*(_QWORD *)a1 + 280LL);
      if ( *(_BYTE *)v27 == 15 || (v27 = *(_QWORD *)(v27 - 8LL * *(unsigned int *)(v27 + 8))) != 0 )
      {
        v30 = v27;
        v31 = -(__int64)*(unsigned int *)(v27 + 8);
        v32 = *(const char **)(v30 + 8 * v31);
        if ( v32 )
          v32 = (const char *)sub_161E970(*(_QWORD *)(v30 + 8 * v31));
        else
          v33 = 0;
      }
      else
      {
        v33 = 0;
        v32 = byte_3F871B3;
      }
      v29(a1, v32, v33, v28, 1);
    }
    v3 = *(_BYTE **)(a1 + 744);
  }
  v4 = *(_QWORD *)(a1 + 792);
  if ( v4 )
  {
    for ( i = *(_QWORD *)(v4 + 8); i; v3 = *(_BYTE **)(a1 + 744) )
    {
      while ( 1 )
      {
        v6 = sub_1648700(i);
        if ( *((_BYTE *)v6 + 16) == 78 && *(_BYTE **)(v6[5] + 56LL) == v3 )
          break;
        i = *(_QWORD *)(i + 8);
        if ( !i )
          goto LABEL_11;
      }
      sub_1263B40((__int64)v56, ".pragma \"coroutine\";\n");
      i = *(_QWORD *)(i + 8);
    }
  }
LABEL_11:
  if ( *(_DWORD *)(*(_QWORD *)(a1 + 232) + 952LL) == 1 )
  {
    sub_214CAD0(v3, (__int64)v56);
    v3 = *(_BYTE **)(a1 + 744);
  }
  if ( (unsigned __int8)sub_1C2F070((__int64)v3) )
    sub_1263B40((__int64)v56, ".entry ");
  else
    sub_1263B40((__int64)v56, ".func ");
  v7 = *(_QWORD *)(a1 + 744);
  if ( (unsigned __int8)sub_1C2FA50(v7) )
    sub_214C940(v7, (__int64)v56);
  sub_214D1D0(
    a1,
    **(_QWORD **)(*(_QWORD *)(**(_QWORD **)(a1 + 264) + 24LL) + 16LL),
    **(_QWORD **)(a1 + 264),
    (__int64)v56);
  sub_38E2490(*(_QWORD *)(a1 + 304), v56, *(_QWORD *)(a1 + 240));
  sub_21502D0((_QWORD *)a1, **(_QWORD **)(a1 + 264), (__int64)v56);
  if ( (unsigned __int8)sub_1C2F070(*(_QWORD *)(a1 + 744)) )
    sub_214DA90(a1, *(_QWORD *)(a1 + 744), (__int64)v56);
  sub_214E300(a1, *(_QWORD *)(a1 + 744), (__int64)v56);
  v8 = *(_QWORD *)(a1 + 744);
  v9 = (v8 >> 2) & 1;
  if ( ((v8 >> 2) & 1) != 0 )
  {
    v49 = *(_QWORD *)(a1 + 744);
    v10 = v8 & 0xFFFFFFFFFFFFFFF8LL;
    v25 = sub_1560260((_QWORD *)((v8 & 0xFFFFFFFFFFFFFFF8LL) + 56), -1, 29);
    v24 = v49;
    if ( v25 )
      goto LABEL_70;
  }
  else
  {
    v10 = v8 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v8 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      if ( !(unsigned __int8)sub_1560180((v8 & 0xFFFFFFFFFFFFFFF8LL) + 112, 29) )
        goto LABEL_22;
      goto LABEL_59;
    }
    v48 = *(_QWORD *)(a1 + 744);
    v23 = sub_1560260((_QWORD *)0x38, -1, 29);
    v24 = v48;
    if ( v23 )
    {
LABEL_51:
      if ( *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(v10 + 64) + 16LL) + 8LL) )
        goto LABEL_22;
      goto LABEL_52;
    }
  }
  v26 = *(_QWORD *)(v10 - 24);
  v50 = v24;
  if ( !*(_BYTE *)(v26 + 16) )
  {
    v54[0] = *(_QWORD *)(v26 + 112);
    if ( (unsigned __int8)sub_1560260(v54, -1, 29) )
    {
      if ( !(_BYTE)v9 )
      {
        if ( (v50 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        {
LABEL_59:
          if ( *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(v10 + 24) + 16LL) + 8LL) )
            goto LABEL_22;
LABEL_52:
          if ( (unsigned __int8)sub_1C2F070(v10) )
            goto LABEL_22;
LABEL_71:
          sub_1263B40((__int64)v56, ".noreturn ");
          goto LABEL_22;
        }
        goto LABEL_51;
      }
LABEL_70:
      if ( *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(v10 + 64) + 16LL) + 8LL) )
        goto LABEL_22;
      goto LABEL_71;
    }
  }
LABEL_22:
  v11 = *(_QWORD *)(a1 + 256);
  v12 = *((unsigned int *)v58 + 2);
  v13 = (__int64 *)*v58;
  LOWORD(v55) = 261;
  v51 = v13;
  v54[0] = (__int64)&v51;
  v52 = v12;
  sub_38DD5A0(v11, v54);
  v14 = *(_DWORD *)(a1 + 824);
  ++*(_QWORD *)(a1 + 808);
  if ( !v14 && !*(_DWORD *)(a1 + 828) )
    goto LABEL_35;
  v15 = *(_QWORD **)(a1 + 816);
  v16 = *(unsigned int *)(a1 + 832);
  v17 = &v15[5 * v16];
  v18 = 4 * v14;
  if ( (unsigned int)(4 * v14) < 0x40 )
    v18 = 64;
  if ( (unsigned int)v16 <= v18 )
  {
    while ( v17 != v15 )
    {
      if ( *v15 != -8 )
      {
        if ( *v15 != -16 )
          j___libc_free_0(v15[2]);
        *v15 = -8;
      }
      v15 += 5;
    }
LABEL_34:
    *(_QWORD *)(a1 + 824) = 0;
    goto LABEL_35;
  }
  do
  {
    if ( *v15 != -16 && *v15 != -8 )
    {
      v46 = v14;
      j___libc_free_0(v15[2]);
      v14 = v46;
    }
    v15 += 5;
  }
  while ( v17 != v15 );
  v36 = *(_DWORD *)(a1 + 832);
  if ( !v14 )
  {
    if ( v36 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 816));
      *(_QWORD *)(a1 + 816) = 0;
      *(_QWORD *)(a1 + 824) = 0;
      *(_DWORD *)(a1 + 832) = 0;
      goto LABEL_35;
    }
    goto LABEL_34;
  }
  v37 = 64;
  v38 = v14 - 1;
  if ( v38 )
  {
    _BitScanReverse(&v38, v38);
    v37 = (unsigned int)(1 << (33 - (v38 ^ 0x1F)));
    if ( (int)v37 < 64 )
      v37 = 64;
  }
  v39 = *(_QWORD **)(a1 + 816);
  if ( (_DWORD)v37 == v36 )
  {
    *(_QWORD *)(a1 + 824) = 0;
    v45 = &v39[5 * v37];
    do
    {
      if ( v39 )
        *v39 = -8;
      v39 += 5;
    }
    while ( v45 != v39 );
  }
  else
  {
    v47 = v37;
    j___libc_free_0(v39);
    v40 = ((((((((4 * v47 / 3u + 1) | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 2)
             | (4 * v47 / 3u + 1)
             | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 4)
           | (((4 * v47 / 3u + 1) | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 2)
           | (4 * v47 / 3u + 1)
           | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v47 / 3u + 1) | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 2)
           | (4 * v47 / 3u + 1)
           | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 4)
         | (((4 * v47 / 3u + 1) | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 2)
         | (4 * v47 / 3u + 1)
         | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 16;
    v41 = (v40
         | (((((((4 * v47 / 3u + 1) | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 2)
             | (4 * v47 / 3u + 1)
             | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 4)
           | (((4 * v47 / 3u + 1) | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 2)
           | (4 * v47 / 3u + 1)
           | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v47 / 3u + 1) | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 2)
           | (4 * v47 / 3u + 1)
           | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 4)
         | (((4 * v47 / 3u + 1) | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 2)
         | (4 * v47 / 3u + 1)
         | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 832) = v41;
    v42 = (_QWORD *)sub_22077B0(40 * v41);
    v43 = *(unsigned int *)(a1 + 832);
    *(_QWORD *)(a1 + 824) = 0;
    *(_QWORD *)(a1 + 816) = v42;
    for ( j = &v42[5 * v43]; j != v42; v42 += 5 )
    {
      if ( v42 )
        *v42 = -8;
    }
  }
LABEL_35:
  v19 = *(_QWORD *)(a1 + 256);
  v51 = (__int64 *)"{\n";
  LOWORD(v55) = 261;
  v54[0] = (__int64)&v51;
  v52 = 2;
  sub_38DD5A0(v19, v54);
  v20 = sub_3936750();
  if ( (unsigned __int8)sub_39371E0(*(_QWORD *)(a1 + 744), v20) )
  {
    v34 = (const char *)sub_3936860(v20, 0);
    sub_214B770(v54, v34);
    v35 = *(_QWORD *)(a1 + 256);
    v53 = 260;
    v51 = v54;
    sub_38DD5A0(v35, &v51);
    if ( (__int64 *)v54[0] != &v55 )
      j_j___libc_free_0(v54[0], v55 + 1);
  }
  sub_39367A0(v20);
  sub_2158E80(a1, *(_QWORD *)(a1 + 264));
  v21 = sub_1626D20(**(_QWORD **)(a1 + 264));
  if ( v21 )
  {
    if ( *(_DWORD *)(*(_QWORD *)(v21 + 8 * (5LL - *(unsigned int *)(v21 + 8))) + 36LL) != 3 )
    {
      v22 = *(_QWORD *)(a1 + 272);
      if ( v22 )
      {
        if ( *(_BYTE *)(v22 + 1744) )
          sub_396E940(a1, *(_QWORD *)(a1 + 264));
      }
    }
  }
  v56[0] = &unk_49EFD28;
  sub_16E7960((__int64)v56);
  if ( (_BYTE *)v59[0] != v60 )
    _libc_free(v59[0]);
}
