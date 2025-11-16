// Function: sub_32416F0
// Address: 0x32416f0
//
void __fastcall sub_32416F0(__int64 a1, __int64 a2)
{
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // r15
  unsigned __int8 v7; // al
  __int64 v8; // rdx
  __int64 v9; // r9
  int v10; // eax
  __int64 v11; // rcx
  _QWORD *v12; // r13
  _BYTE *v13; // rdx
  __int64 v14; // rax
  __int64 v15; // r8
  __int64 v16; // rsi
  bool v17; // di
  __int64 v18; // r10
  __int64 *v19; // r15
  int v20; // ecx
  unsigned int v21; // eax
  __int64 *v22; // rbx
  __int64 v23; // r11
  _BYTE *v24; // rcx
  int v25; // eax
  int v26; // r8d
  __int64 v27; // rdx
  __int64 v28; // rdi
  _BYTE *v29; // rdx
  __int64 (*v30)(); // rcx
  __int64 v31; // rax
  unsigned int v32; // r15d
  unsigned int v33; // eax
  __int64 v34; // r9
  int v35; // eax
  __int64 v36; // rax
  __int64 (*v37)(); // rax
  __int64 v38; // rdi
  char (__fastcall *v39)(__int64, __int64); // rax
  int v40; // eax
  __int64 v41; // rax
  __int64 v42; // rax
  unsigned int v43; // edx
  unsigned int v44; // eax
  __int64 v45; // rax
  _BYTE *v46; // rdx
  __int64 v47; // rdi
  __int64 (*v48)(); // rcx
  __int64 v49; // rax
  _BYTE *v50; // r15
  unsigned int v51; // r12d
  unsigned int v52; // eax
  __int64 v53; // r9
  __int64 v54; // rsi
  __int64 v55; // rsi
  int v56; // ebx
  int v57; // eax
  char v58; // al
  __int64 v59; // [rsp+0h] [rbp-140h]
  __int64 v60; // [rsp+8h] [rbp-138h]
  int v61; // [rsp+10h] [rbp-130h]
  int v62; // [rsp+10h] [rbp-130h]
  _BYTE *v63; // [rsp+10h] [rbp-130h]
  __int64 v64; // [rsp+10h] [rbp-130h]
  __int64 v65; // [rsp+18h] [rbp-128h]
  unsigned int v66; // [rsp+28h] [rbp-118h]
  __int64 v67; // [rsp+28h] [rbp-118h]
  void *v68; // [rsp+30h] [rbp-110h] BYREF
  __int64 v69; // [rsp+38h] [rbp-108h] BYREF
  __int64 v70; // [rsp+40h] [rbp-100h]
  __int64 v71; // [rsp+48h] [rbp-F8h]
  __int64 v72; // [rsp+50h] [rbp-F0h]
  __int64 v73; // [rsp+58h] [rbp-E8h]
  _BYTE **v74; // [rsp+60h] [rbp-E0h]
  _BYTE *v75; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v76; // [rsp+78h] [rbp-C8h]
  __int64 v77; // [rsp+80h] [rbp-C0h]
  _BYTE v78[184]; // [rsp+88h] [rbp-B8h] BYREF

  v4 = sub_2E88D60(a2);
  v5 = sub_B92180(*(_QWORD *)v4);
  if ( !v5
    || ((v6 = v5, v7 = *(_BYTE *)(v5 - 16), (v7 & 2) != 0)
      ? (v8 = *(_QWORD *)(v6 - 32))
      : (v8 = v6 - 16 - 8LL * ((v7 >> 2) & 0xF)),
        !*(_DWORD *)(*(_QWORD *)(v8 + 40) + 32LL)) )
  {
    sub_3211700((__int64 *)a1, a2);
    return;
  }
  if ( (*(_BYTE *)(v6 + 35) & 0x20) != 0 && sub_2E88ED0(a2, 1) )
  {
    v35 = *(_DWORD *)(a2 + 44);
    if ( (v35 & 4) != 0 || (v35 & 8) == 0 )
      v36 = (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) >> 17) & 1LL;
    else
      LOBYTE(v36) = sub_2E88A90(a2, 0x20000, 1);
    if ( !(_BYTE)v36 || (*(_BYTE *)(a2 + 44) & 8) != 0 )
    {
      v37 = *(__int64 (**)())(**(_QWORD **)(v4 + 16) + 128LL);
      if ( v37 == sub_2DAC790 )
        BUG();
      v38 = v37();
      v39 = *(char (__fastcall **)(__int64, __int64))(*(_QWORD *)v38 + 1328LL);
      if ( v39 == sub_2FDE950 )
      {
        v40 = *(_DWORD *)(a2 + 44);
        if ( (v40 & 4) != 0 || (v40 & 8) == 0 )
          v41 = (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) >> 5) & 1LL;
        else
          LOBYTE(v41) = sub_2E88A90(a2, 32, 1);
        if ( !(_BYTE)v41 )
        {
LABEL_48:
          v68 = (void *)a2;
          v69 = 0;
          sub_3228BB0((__int64)&v75, a1 + 464, (__int64 *)&v68, &v69);
          goto LABEL_6;
        }
        v57 = *(_DWORD *)(a2 + 44);
        if ( (v57 & 4) != 0 || (v57 & 8) == 0 )
          v58 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) >> 7;
        else
          v58 = sub_2E88A90(a2, 128, 1);
      }
      else
      {
        v58 = v39(v38, a2);
      }
      if ( v58 )
      {
        v68 = (void *)a2;
        v69 = 0;
        sub_3228BB0((__int64)&v75, a1 + 432, (__int64 *)&v68, &v69);
      }
      goto LABEL_48;
    }
  }
LABEL_6:
  sub_3211700((__int64 *)a1, a2);
  if ( !*(_QWORD *)(a1 + 64) )
    return;
  if ( (*(_BYTE *)(*(_QWORD *)(a2 + 16) + 24LL) & 0x10) != 0 )
    return;
  v10 = *(_DWORD *)(a2 + 44);
  if ( (v10 & 1) != 0 )
    return;
  v11 = *(_QWORD *)(a1 + 8);
  v12 = (_QWORD *)(a2 + 56);
  if ( (v10 & 2) == 0 )
  {
    v66 = 0;
LABEL_17:
    v13 = *(_BYTE **)(a2 + 56);
    v15 = *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(v11 + 224) + 8LL) + 1780LL);
    if ( v13 )
      goto LABEL_18;
    goto LABEL_59;
  }
  v13 = *(_BYTE **)(a2 + 56);
  if ( v13 )
  {
    v14 = *(_QWORD *)(a2 + 24);
    if ( !v14 || *(_QWORD *)(a1 + 56) == v14 )
    {
      v66 = 0;
      v15 = *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(v11 + 224) + 8LL) + 1780LL);
      goto LABEL_18;
    }
    *(_QWORD *)(a1 + 56) = v14;
    v66 = 8;
    goto LABEL_17;
  }
  v66 = 0;
  v15 = *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(v11 + 224) + 8LL) + 1780LL);
LABEL_59:
  if ( *(_QWORD *)(a1 + 48) == a2 )
  {
    sub_3221AB0(a1, *(_DWORD *)(v6 + 20), 0, (_BYTE *)v6, 5u, v9, 0, 0);
    return;
  }
  v13 = 0;
LABEL_18:
  v16 = *(_QWORD *)(a1 + 40);
  v17 = 1;
  if ( v16 )
    v17 = *(_DWORD *)(v16 + 252) == *(_DWORD *)(*(_QWORD *)(a2 + 24) + 252LL)
       && *(_DWORD *)(*(_QWORD *)(a2 + 24) + 256LL) == *(_DWORD *)(v16 + 256);
  if ( (*(_BYTE *)(a1 + 3008) & 1) != 0 )
  {
    v18 = a1 + 3016;
    v19 = (__int64 *)(a1 + 3048);
    v20 = 3;
  }
  else
  {
    v42 = *(unsigned int *)(a1 + 3024);
    v18 = *(_QWORD *)(a1 + 3016);
    v19 = (__int64 *)(v18 + 8 * v42);
    if ( !(_DWORD)v42 )
    {
LABEL_84:
      v22 = v19;
      goto LABEL_23;
    }
    v20 = v42 - 1;
  }
  v21 = v20 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v22 = (__int64 *)(v18 + 8LL * v21);
  v23 = *v22;
  if ( a2 != *v22 )
  {
    v56 = 1;
    while ( v23 != -4096 )
    {
      v9 = (unsigned int)(v56 + 1);
      v21 = v20 & (v56 + v21);
      v22 = (__int64 *)(v18 + 8LL * v21);
      v23 = *v22;
      if ( a2 == *v22 )
        goto LABEL_23;
      v56 = v9;
    }
    goto LABEL_84;
  }
LABEL_23:
  v24 = *(_BYTE **)(a1 + 24);
  if ( v24 == v13 && v22 == v19 && v17 )
  {
    if ( v13 && (!(_DWORD)v15 && (unsigned int)sub_B10CE0(a2 + 56) || v66) )
    {
      v45 = *(_QWORD *)(a1 + 8);
      v76 = 0;
      v75 = v78;
      v46 = v78;
      v77 = 128;
      v47 = *(_QWORD *)(v45 + 224);
      v48 = *(__int64 (**)())(*(_QWORD *)v47 + 96LL);
      v49 = 0;
      if ( v48 != sub_C13EE0 )
      {
        if ( ((unsigned __int8 (__fastcall *)(__int64, __int64, _BYTE *, __int64 (*)(), __int64, __int64))v48)(
               v47,
               v16,
               v78,
               v48,
               v15,
               v9) )
        {
          v73 = 0x100000000LL;
          v74 = &v75;
          v69 = 2;
          v68 = &unk_49DD288;
          v70 = 0;
          v71 = 0;
          v72 = 0;
          sub_CB5980((__int64)&v68, 0, 0, 0);
          sub_B10EE0(v12, (__int64)&v68);
          v68 = &unk_49DD388;
          sub_CB5840((__int64)&v68);
        }
        v49 = v76;
        v46 = v75;
      }
      v64 = (__int64)v46;
      v65 = v49;
      v50 = (_BYTE *)sub_B10D00((__int64)v12);
      v51 = sub_B10CF0((__int64)v12);
      v52 = sub_B10CE0((__int64)v12);
      sub_3221AB0(a1, v52, v51, v50, v66, v53, v64, v65);
      if ( v75 != v78 )
        _libc_free((unsigned __int64)v75);
    }
  }
  else if ( v13 )
  {
    v61 = v15;
    v25 = sub_B10CE0(a2 + 56);
    v26 = v61;
    if ( v61 | v25 )
    {
      if ( *(_QWORD *)(a1 + 48) == a2 )
      {
        v66 |= 5u;
        *(_QWORD *)(a1 + 48) = 0;
      }
      if ( *(_QWORD *)(a1 + 24) )
        v26 = sub_B10CE0(a1 + 24);
      v62 = v26;
      if ( (unsigned int)sub_B10CE0(a2 + 56) && ((unsigned int)sub_B10CE0(a2 + 56) != v62 || v22 != v19) )
        v66 |= 1u;
      v27 = *(_QWORD *)(a1 + 8);
      if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(v27 + 200) + 544LL) - 42) <= 1 && *(_DWORD *)(a1 + 6224) == 1 )
      {
        sub_322A990(a1, a2, v66);
      }
      else
      {
        v76 = 0;
        v75 = v78;
        v77 = 128;
        v28 = *(_QWORD *)(v27 + 224);
        v29 = v78;
        v30 = *(__int64 (**)())(*(_QWORD *)v28 + 96LL);
        v31 = 0;
        if ( v30 != sub_C13EE0 )
        {
          if ( ((unsigned __int8 (__fastcall *)(__int64, __int64, _BYTE *))v30)(v28, v16, v78) )
          {
            v73 = 0x100000000LL;
            v74 = &v75;
            v69 = 2;
            v68 = &unk_49DD288;
            v70 = 0;
            v71 = 0;
            v72 = 0;
            sub_CB5980((__int64)&v68, 0, 0, 0);
            sub_B10EE0((_QWORD *)(a2 + 56), (__int64)&v68);
            v68 = &unk_49DD388;
            sub_CB5840((__int64)&v68);
          }
          v31 = v76;
          v29 = v75;
        }
        v59 = (__int64)v29;
        v60 = v31;
        v63 = (_BYTE *)sub_B10D00(a2 + 56);
        v32 = sub_B10CF0(a2 + 56);
        v33 = sub_B10CE0(a2 + 56);
        sub_3221AB0(a1, v33, v32, v63, v66, v34, v59, v60);
        if ( v75 != v78 )
          _libc_free((unsigned __int64)v75);
      }
      if ( (unsigned int)sub_B10CE0(a2 + 56) && v12 != (_QWORD *)(a1 + 24) )
      {
        v54 = *(_QWORD *)(a1 + 24);
        if ( v54 )
          sub_B91220(a1 + 24, v54);
        v55 = *(_QWORD *)(a2 + 56);
        *(_QWORD *)(a1 + 24) = v55;
        if ( v55 )
          sub_B96E90(a1 + 24, v55, 1);
      }
    }
  }
  else if ( (_DWORD)v15
         && dword_5037528 != 2
         && (dword_5037528 == 1 || *(_QWORD *)(a1 + 32) || v16 && v16 != *(_QWORD *)(a2 + 24)) )
  {
    v43 = 0;
    if ( v24 )
    {
      v67 = sub_B10D00(a1 + 24);
      v44 = sub_B10CF0(a1 + 24);
      v24 = (_BYTE *)v67;
      v43 = v44;
    }
    sub_3221AB0(a1, 0, v43, v24, 0, v9, 0, 0);
  }
}
