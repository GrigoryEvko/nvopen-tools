// Function: sub_37394D0
// Address: 0x37394d0
//
unsigned __int8 *__fastcall sub_37394D0(__int64 *a1, unsigned __int8 *a2, __int64 a3)
{
  char v5; // al
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  unsigned __int8 *v9; // r12
  __int64 v10; // rax
  __int64 v11; // r15
  __int64 v12; // r11
  unsigned __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // r10
  _QWORD *v16; // r8
  __int64 v17; // rax
  char v18; // al
  __int64 (*v19)(); // rax
  __int64 v20; // rax
  int v21; // eax
  unsigned __int64 v22; // rdx
  __int64 v24; // rax
  char *v25; // rdi
  size_t v26; // rdx
  __int64 v27; // rax
  unsigned __int64 **v28; // r15
  __int64 *v29; // rsi
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rcx
  __int64 v33; // [rsp+8h] [rbp-128h]
  __int64 v34; // [rsp+10h] [rbp-120h]
  __int64 v35; // [rsp+18h] [rbp-118h]
  unsigned int v36; // [rsp+18h] [rbp-118h]
  unsigned int v37; // [rsp+20h] [rbp-110h]
  unsigned __int64 v39; // [rsp+28h] [rbp-108h]
  unsigned __int64 *v40[2]; // [rsp+30h] [rbp-100h] BYREF
  void *src; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v42; // [rsp+48h] [rbp-E8h]
  _BYTE v43[32]; // [rsp+50h] [rbp-E0h] BYREF
  char *v44; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v45; // [rsp+78h] [rbp-B8h]
  char dest[8]; // [rsp+80h] [rbp-B0h] BYREF
  char *v47; // [rsp+88h] [rbp-A8h]
  char v48; // [rsp+98h] [rbp-98h] BYREF
  __int64 **v49; // [rsp+E0h] [rbp-50h]

  v5 = sub_3736590(a1);
  src = v43;
  v9 = sub_3250680(a1, a2, v5);
  v10 = a1[23];
  v42 = 0x200000000LL;
  v11 = *(_QWORD *)(v10 + 336);
  v12 = v11 + 24LL * *(unsigned int *)(v10 + 344);
  if ( v11 == v12 )
  {
    v45 = 0x200000000LL;
    v44 = dest;
  }
  else
  {
    v13 = 2;
    v7 = 0;
    while ( 1 )
    {
      v8 = v7 + 1;
      v14 = *(_QWORD *)(v11 + 16);
      v15 = *(_QWORD *)(v11 + 8);
      if ( v7 + 1 > v13 )
      {
        v33 = *(_QWORD *)(v11 + 8);
        v34 = v12;
        v35 = *(_QWORD *)(v11 + 16);
        sub_C8D5F0((__int64)&src, v43, v7 + 1, 0x10u, v7, v8);
        v7 = (unsigned int)v42;
        v15 = v33;
        v12 = v34;
        v14 = v35;
      }
      v16 = (char *)src + 16 * v7;
      v11 += 24;
      *v16 = v15;
      v16[1] = v14;
      v7 = (unsigned int)(v42 + 1);
      LODWORD(v42) = v42 + 1;
      if ( v11 == v12 )
        break;
      v13 = HIDWORD(v42);
    }
    v44 = dest;
    v45 = 0x200000000LL;
    if ( (_DWORD)v7 )
    {
      v25 = dest;
      v26 = 16LL * (unsigned int)v7;
      if ( (unsigned int)v7 <= 2
        || (v36 = v7,
            sub_C8D5F0((__int64)&v44, dest, (unsigned int)v7, 0x10u, v7, v8),
            v25 = v44,
            v7 = v36,
            (v26 = 16LL * (unsigned int)v42) != 0) )
      {
        v37 = v7;
        memcpy(v25, src, v26);
        v7 = v37;
      }
      LODWORD(v45) = v7;
    }
  }
  sub_3739060(a1, (__int64)v9, (char *)&v44, v6, v7, v8);
  if ( v44 != dest )
    _libc_free((unsigned __int64)v44);
  v17 = a1[26];
  if ( *(_BYTE *)(v17 + 3768)
    && !(unsigned __int8)sub_35DDD70(*(_QWORD *)(*(_QWORD *)(v17 + 3048) + 8LL) + 856LL, *(__int64 **)(v17 + 3048)) )
  {
    sub_3249FA0(a1, (__int64)v9, 16359);
  }
  v18 = sub_3737600();
  if ( a3 && v18 )
  {
    v24 = sub_31DA6B0(a1[23]);
    sub_324AC60(a1, (__int64)v9, 15884, a3, *(_QWORD *)(*(_QWORD *)(v24 + 96) + 16LL));
  }
  if ( !sub_3736590(a1) )
  {
    v19 = *(__int64 (**)())(**(_QWORD **)(*(_QWORD *)(a1[23] + 232) + 16LL) + 136LL);
    if ( v19 == sub_2DD19D0 )
      BUG();
    v20 = v19();
    v21 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v20 + 368LL))(v20, *(_QWORD *)(a1[23] + 232));
    v39 = v22;
    if ( v21 == 1 )
    {
      v27 = sub_A777F0(0x10u, a1 + 11);
      v28 = (unsigned __int64 **)v27;
      if ( v27 )
      {
        *(_QWORD *)v27 = 0;
        *(_DWORD *)(v27 + 8) = 0;
      }
      sub_3249B00(a1, (unsigned __int64 **)v27, 11, 156);
      if ( !v39 )
        goto LABEL_34;
      sub_3249B00(a1, v28, 11, 17);
      LODWORD(v44) = 65549;
      sub_32499D0(a1, v28, 65549, v39);
      v32 = 34;
    }
    else
    {
      if ( v21 != 2 )
      {
        if ( !v21 && (unsigned int)(v22 - 1) <= 0x3FFFFFFE )
        {
          LOBYTE(v44) = 1;
          HIDWORD(v44) = v22;
          sub_3738310(a1, (__int64)v9, 64, (__int64)&v44);
        }
        goto LABEL_19;
      }
      v29 = a1 + 11;
      if ( (_DWORD)v22 != 3 )
      {
        v30 = sub_A777F0(0x10u, v29);
        if ( v30 )
        {
          *(_QWORD *)v30 = 0;
          *(_DWORD *)(v30 + 8) = 0;
        }
        sub_3247620((__int64)&v44, a1[23], (__int64)a1, v30);
        v40[0] = 0;
        v40[1] = 0;
        sub_3243F80(&v44, v39, HIDWORD(v39));
        sub_3244870(&v44, v40);
        sub_3243D40((__int64)&v44);
        sub_3249620(a1, (__int64)v9, 64, v49);
        if ( v47 != &v48 )
          _libc_free((unsigned __int64)v47);
        goto LABEL_19;
      }
      v31 = sub_A777F0(0x10u, v29);
      v28 = (unsigned __int64 **)v31;
      if ( v31 )
      {
        *(_QWORD *)v31 = 0;
        *(_DWORD *)(v31 + 8) = 0;
      }
      sub_3735D90(a1, (unsigned __int64 **)v31, (char)"__stack_pointer", 15, HIDWORD(v39));
      v32 = 159;
    }
    sub_3249B00(a1, v28, 11, v32);
LABEL_34:
    sub_3249620(a1, (__int64)v9, 64, (__int64 **)v28);
  }
LABEL_19:
  sub_32379A0(a1[26], (__int64)a1, *(_DWORD *)(a1[10] + 36), (__int64)a2, (__int64)v9);
  if ( src != v43 )
    _libc_free((unsigned __int64)src);
  return v9;
}
