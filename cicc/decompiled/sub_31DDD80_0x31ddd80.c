// Function: sub_31DDD80
// Address: 0x31ddd80
//
void __fastcall sub_31DDD80(__int64 a1, __int64 a2, unsigned int *a3, __int64 a4, char a5)
{
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdi
  void (*v18)(); // rax
  __int64 v19; // rbx
  __int64 v20; // rax
  __int64 *v21; // r13
  __int64 *v22; // r12
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 (*v26)(); // rax
  __int64 v27; // rax
  __int64 *v28; // r14
  __int64 *v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  char v33; // al
  __int64 v34; // rsi
  _QWORD *v35; // r12
  __int64 v36; // rax
  unsigned __int64 v37; // rax
  __int64 v38; // r12
  void (__fastcall *v39)(__int64, __int64, unsigned __int64); // rbx
  __int64 v40; // rax
  __int64 v41; // r13
  __int64 *v42; // rax
  __int64 v43; // rdi
  void (*v44)(); // rax
  __int64 v45; // rax
  __int64 v46; // [rsp+0h] [rbp-130h]
  __int64 *v47; // [rsp+8h] [rbp-128h]
  __int64 v48; // [rsp+18h] [rbp-118h]
  int v49; // [rsp+24h] [rbp-10Ch]
  unsigned __int64 v50; // [rsp+30h] [rbp-100h]
  __int64 v51; // [rsp+30h] [rbp-100h]
  __int64 v52; // [rsp+38h] [rbp-F8h]
  __int64 *v53; // [rsp+40h] [rbp-F0h]
  void (__fastcall *v54)(__int64, __int64, _QWORD); // [rsp+40h] [rbp-F0h]
  unsigned int *v57; // [rsp+50h] [rbp-E0h]
  unsigned int *v58; // [rsp+58h] [rbp-D8h]
  __int64 v59; // [rsp+60h] [rbp-D0h] BYREF
  __int64 *v60; // [rsp+68h] [rbp-C8h]
  __int64 v61; // [rsp+70h] [rbp-C0h]
  int v62; // [rsp+78h] [rbp-B8h]
  char v63; // [rsp+7Ch] [rbp-B4h]
  char v64; // [rsp+80h] [rbp-B0h] BYREF

  if ( !a4 )
    return;
  v9 = sub_31DA6B0(a1);
  v10 = *(_QWORD *)(a1 + 200);
  v11 = **(_QWORD **)(a1 + 232);
  if ( (*(_BYTE *)(v10 + 879) & 2) != 0 )
    v12 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v9 + 104LL))(
            v9,
            v11,
            v10,
            *(_QWORD *)(a2 + 8) + 32LL * *a3);
  else
    v12 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v9 + 96LL))(v9, v11);
  v52 = sub_2E79000(*(__int64 **)(a1 + 232));
  if ( a5 )
  {
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 224) + 176LL))(*(_QWORD *)(a1 + 224), v12, 0);
    v13 = sub_2E79000(*(__int64 **)(a1 + 232));
    LODWORD(v14) = sub_2E79B10((_DWORD *)a2, v13);
    if ( (_DWORD)v14 )
    {
      _BitScanReverse64((unsigned __int64 *)&v14, (unsigned int)v14);
      sub_31DCA70(a1, 63 - (v14 ^ 0x3F), 0, 0);
    }
    else
    {
      sub_31DCA70(a1, 0xFFFFFFFF, 0, 0);
    }
  }
  else
  {
    v15 = sub_2E79000(*(__int64 **)(a1 + 232));
    LODWORD(v16) = sub_2E79B10((_DWORD *)a2, v15);
    if ( (_DWORD)v16 )
    {
      _BitScanReverse64((unsigned __int64 *)&v16, (unsigned int)v16);
      sub_31DCA70(a1, 63 - (v16 ^ 0x3F), 0, 0);
    }
    else
    {
      sub_31DCA70(a1, 0xFFFFFFFF, 0, 0);
    }
    v17 = *(_QWORD *)(a1 + 224);
    v18 = *(void (**)())(*(_QWORD *)v17 + 240LL);
    if ( v18 != nullsub_101 )
      ((void (__fastcall *)(__int64, __int64))v18)(v17, 3);
  }
  v58 = a3;
  v57 = &a3[a4];
  if ( a3 == v57 )
    goto LABEL_40;
  while ( 2 )
  {
    v19 = *v58;
    v20 = *(_QWORD *)(a2 + 8) + 32 * v19;
    v21 = *(__int64 **)v20;
    v22 = *(__int64 **)(v20 + 8);
    if ( v22 == *(__int64 **)v20 )
      goto LABEL_15;
    if ( *(_DWORD *)a2 != 3 || !*(_BYTE *)(*(_QWORD *)(a1 + 208) + 280LL) )
      goto LABEL_18;
    v63 = 1;
    v61 = 16;
    v60 = (__int64 *)&v64;
    v25 = *(_QWORD *)(a1 + 232);
    v62 = 0;
    v59 = 0;
    v26 = *(__int64 (**)())(**(_QWORD **)(v25 + 16) + 144LL);
    if ( v26 == sub_2C8F680 )
      BUG();
    v27 = v26();
    v46 = a2;
    v28 = v21;
    v48 = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD, _QWORD))(*(_QWORD *)v27 + 1944LL))(
            v27,
            *(_QWORD *)(a1 + 232),
            (unsigned int)v19,
            *(_QWORD *)(a1 + 216));
    v33 = v63;
    v53 = v22;
    v47 = v21;
    v49 = v19;
    do
    {
      v41 = *v28;
      if ( v33 )
      {
        v30 = (__int64)v60;
        v29 = &v60[HIDWORD(v61)];
        while ( 1 )
        {
          v42 = v60;
          if ( v60 == v29 )
            break;
          while ( *v42 != v41 )
          {
            if ( v29 == ++v42 )
              goto LABEL_45;
          }
          if ( v53 == ++v28 )
          {
            v22 = v53;
            v21 = v47;
            LODWORD(v19) = v49;
            a2 = v46;
            goto LABEL_18;
          }
          v41 = *v28;
        }
LABEL_45:
        if ( HIDWORD(v61) >= (unsigned int)v61 )
          goto LABEL_29;
        v34 = (unsigned int)++HIDWORD(v61);
        *v29 = v41;
        ++v59;
      }
      else
      {
LABEL_29:
        v34 = v41;
        sub_C8CC70((__int64)&v59, v41, (__int64)v29, v30, v31, v32);
        v33 = v63;
        if ( !(_BYTE)v29 )
          goto LABEL_31;
      }
      v35 = *(_QWORD **)(a1 + 216);
      v36 = sub_2E309C0(v41, v34, (__int64)v29, v30, v31);
      v37 = sub_E808D0(v36, 0, v35, 0);
      v38 = *(_QWORD *)(a1 + 224);
      v39 = *(void (__fastcall **)(__int64, __int64, unsigned __int64))(*(_QWORD *)v38 + 272LL);
      v50 = sub_E81A00(18, v37, v48, *(_QWORD **)(a1 + 216), 0);
      v40 = sub_31DD620(a1, v49, *(_DWORD *)(v41 + 24));
      v39(v38, v40, v50);
      v33 = v63;
LABEL_31:
      ++v28;
    }
    while ( v53 != v28 );
    v22 = v53;
    v21 = v47;
    LODWORD(v19) = v49;
    a2 = v46;
    if ( !v33 )
      _libc_free((unsigned __int64)v60);
LABEL_18:
    if ( a5 && *(_DWORD *)(v52 + 24) == 2 )
    {
      v51 = *(_QWORD *)(a1 + 224);
      v54 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v51 + 208LL);
      v45 = sub_31DD380(a1, v19, 1);
      v54(v51, v45, 0);
    }
    v23 = sub_31DD380(a1, v19, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 224) + 208LL))(*(_QWORD *)(a1 + 224), v23, 0);
    do
    {
      v24 = *v21++;
      (*(void (__fastcall **)(__int64, __int64, __int64, _QWORD))(*(_QWORD *)a1 + 600LL))(
        a1,
        a2,
        v24,
        (unsigned int)v19);
    }
    while ( v22 != v21 );
LABEL_15:
    if ( v57 != ++v58 )
      continue;
    break;
  }
LABEL_40:
  if ( (_BYTE)qword_5035EE8 )
    sub_31DD3A0(a1, a2, **(_QWORD **)(a1 + 232));
  if ( !a5 )
  {
    v43 = *(_QWORD *)(a1 + 224);
    v44 = *(void (**)())(*(_QWORD *)v43 + 240LL);
    if ( v44 != nullsub_101 )
      ((void (__fastcall *)(__int64, __int64))v44)(v43, 4);
  }
}
