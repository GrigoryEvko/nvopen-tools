// Function: sub_24BF9E0
// Address: 0x24bf9e0
//
__int64 __fastcall sub_24BF9E0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 *v4; // r13
  unsigned __int64 **v5; // r14
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // r9
  __int64 v15; // r8
  unsigned __int64 v16; // rax
  int v17; // ecx
  unsigned __int64 *v18; // rdx
  char *v19; // rax
  unsigned __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rsi
  __int64 v25; // rax
  __int64 v26; // rsi
  __int64 v27; // rax
  __int64 v28; // r10
  __int64 v29; // r9
  __int64 v30; // rdi
  __int64 i; // r12
  __int64 v32; // r9
  __int64 v33; // r14
  __int64 v34; // rbx
  __int64 *v35; // rax
  __int64 v36; // r13
  __int64 *v37; // r10
  __int64 v38; // rax
  unsigned __int64 **v39; // r9
  unsigned __int64 v40; // rsi
  unsigned __int64 v41; // rax
  __int64 v42; // [rsp+8h] [rbp-180h]
  unsigned __int64 v44; // [rsp+18h] [rbp-170h]
  unsigned __int64 v45; // [rsp+18h] [rbp-170h]
  __int64 v46; // [rsp+20h] [rbp-168h]
  __int64 v47; // [rsp+20h] [rbp-168h]
  __int64 v48; // [rsp+20h] [rbp-168h]
  __int64 *v49; // [rsp+28h] [rbp-160h]
  __int64 v50; // [rsp+30h] [rbp-158h]
  __int64 v51; // [rsp+40h] [rbp-148h]
  char *v52[2]; // [rsp+48h] [rbp-140h] BYREF
  __int64 v53; // [rsp+58h] [rbp-130h] BYREF
  __int64 v54[3]; // [rsp+68h] [rbp-120h] BYREF
  char v55; // [rsp+84h] [rbp-104h]
  __int16 v56; // [rsp+88h] [rbp-100h]
  unsigned __int64 v57; // [rsp+A0h] [rbp-E8h]
  char v58; // [rsp+B4h] [rbp-D4h]
  unsigned __int64 *v59; // [rsp+C8h] [rbp-C0h] BYREF
  unsigned __int64 v60; // [rsp+D0h] [rbp-B8h]
  _BYTE v61[32]; // [rsp+D8h] [rbp-B0h] BYREF
  __int64 v62; // [rsp+F8h] [rbp-90h]
  unsigned __int64 v63; // [rsp+100h] [rbp-88h]
  __int16 v64; // [rsp+108h] [rbp-80h]
  __int64 v65; // [rsp+110h] [rbp-78h]
  void **v66; // [rsp+118h] [rbp-70h]
  _QWORD *v67; // [rsp+120h] [rbp-68h]
  __int64 v68; // [rsp+128h] [rbp-60h]
  int v69; // [rsp+130h] [rbp-58h]
  __int16 v70; // [rsp+134h] [rbp-54h]
  char v71; // [rsp+136h] [rbp-52h]
  __int64 v72; // [rsp+138h] [rbp-50h]
  __int64 v73; // [rsp+140h] [rbp-48h]
  void *v74; // [rsp+148h] [rbp-40h] BYREF
  _QWORD v75[7]; // [rsp+150h] [rbp-38h] BYREF

  v4 = v54;
  v5 = &v59;
  v54[0] = a3;
  sub_2A41C90(
    (unsigned int)&v59,
    a3,
    (unsigned int)"rtsan.module_ctor",
    17,
    (unsigned int)"__rtsan_ensure_initialized",
    26,
    0,
    0,
    0,
    0,
    (__int64)sub_24BF270,
    (__int64)v54,
    0,
    0,
    0);
  v51 = a3 + 24;
  if ( *(_QWORD *)(a3 + 32) != a3 + 24 )
  {
    v7 = *(_QWORD *)(a3 + 32);
    while ( 1 )
    {
      v8 = v7 - 56;
      if ( !v7 )
        v8 = 0;
      if ( (unsigned __int8)sub_B2D610(v8, 61) )
      {
        v25 = *(_QWORD *)(v8 + 80);
        if ( !v25 )
          BUG();
        v26 = *(_QWORD *)(v25 + 32);
        if ( v26 )
          v26 -= 24;
        sub_24BF2E0(v8, v26, "__rtsan_realtime_enter", 0, 0);
        v27 = *(_QWORD *)(v8 + 80);
        v28 = v8 + 72;
        if ( v8 + 72 == v27 )
        {
          v29 = 0;
        }
        else
        {
          if ( !v27 )
            BUG();
          while ( 1 )
          {
            v29 = *(_QWORD *)(v27 + 32);
            if ( v29 != v27 + 24 )
              break;
            v27 = *(_QWORD *)(v27 + 8);
            if ( v28 == v27 )
              break;
            if ( !v27 )
              BUG();
          }
        }
        v30 = v8;
        i = v29;
        v32 = (__int64)v5;
        v33 = v7;
        v34 = v27;
        v35 = v4;
        v36 = v28;
        v37 = v35;
LABEL_40:
        while ( v36 != v34 )
        {
          if ( !i )
            BUG();
          if ( *(_BYTE *)(i - 24) == 30 )
          {
            v49 = v37;
            v50 = v32;
            sub_24BF2E0(v30, i - 24, "__rtsan_realtime_exit", 0, 0);
            v37 = v49;
            v32 = v50;
          }
          for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v34 + 32) )
          {
            v38 = v34 - 24;
            if ( !v34 )
              v38 = 0;
            if ( i != v38 + 48 )
              break;
            v34 = *(_QWORD *)(v34 + 8);
            if ( v36 == v34 )
              goto LABEL_40;
            if ( !v34 )
              BUG();
          }
        }
        v8 = v30;
        v7 = v33;
        v4 = v37;
        sub_24BF280(v32);
        v5 = v39;
        if ( !BYTE4(v65) )
          _libc_free(v63);
        if ( !v61[12] )
          _libc_free(v60);
      }
      if ( !(unsigned __int8)sub_B2D610(v8, 62) )
        goto LABEL_3;
      v9 = *(_QWORD *)(v8 + 80);
      if ( !v9 )
        BUG();
      v10 = *(_QWORD *)(v9 + 32);
      if ( !v10 )
      {
        v3 = sub_BD5C60(0);
        v71 = 7;
        v65 = v3;
        v66 = &v74;
        v67 = v75;
        v59 = (unsigned __int64 *)v61;
        v60 = 0x200000000LL;
        v74 = &unk_49DA100;
        v68 = 0;
        v69 = 0;
        v70 = 512;
        v72 = 0;
        v73 = 0;
        v62 = 0;
        v63 = 0;
        v64 = 0;
        v75[0] = &unk_49DA0B0;
        BUG();
      }
      v44 = *(_QWORD *)(v9 + 32);
      v46 = v10 - 24;
      v11 = sub_BD5C60(v10 - 24);
      v68 = 0;
      v65 = v11;
      v66 = &v74;
      v67 = v75;
      v59 = (unsigned __int64 *)v61;
      v74 = &unk_49DA100;
      v62 = 0;
      v63 = 0;
      v60 = 0x200000000LL;
      v69 = 0;
      v70 = 512;
      v71 = 7;
      v72 = 0;
      v73 = 0;
      v64 = 0;
      v75[0] = &unk_49DA0B0;
      v12 = *(_QWORD *)(v44 + 16);
      v63 = v44;
      v62 = v12;
      v13 = *(_QWORD *)sub_B46C60(v46);
      v54[0] = v13;
      if ( !v13 )
        break;
      sub_B96E90((__int64)v4, v13, 1);
      v15 = v54[0];
      if ( !v54[0] )
        break;
      v16 = (unsigned __int64)v59;
      v17 = v60;
      v18 = &v59[2 * (unsigned int)v60];
      if ( v59 == v18 )
      {
LABEL_56:
        if ( (unsigned int)v60 >= (unsigned __int64)HIDWORD(v60) )
        {
          v40 = (unsigned int)v60 + 1LL;
          v41 = v42 & 0xFFFFFFFF00000000LL;
          v42 &= 0xFFFFFFFF00000000LL;
          if ( HIDWORD(v60) < v40 )
          {
            v45 = v41;
            v48 = v54[0];
            sub_C8D5F0((__int64)v5, v61, v40, 0x10u, v54[0], v14);
            v41 = v45;
            v15 = v48;
            v18 = &v59[2 * (unsigned int)v60];
          }
          *v18 = v41;
          v18[1] = v15;
          v15 = v54[0];
          LODWORD(v60) = v60 + 1;
        }
        else
        {
          if ( v18 )
          {
            *(_DWORD *)v18 = 0;
            v18[1] = v15;
            v17 = v60;
            v15 = v54[0];
          }
          LODWORD(v60) = v17 + 1;
        }
LABEL_60:
        if ( !v15 )
          goto LABEL_18;
        goto LABEL_17;
      }
      while ( *(_DWORD *)v16 )
      {
        v16 += 16LL;
        if ( v18 == (unsigned __int64 *)v16 )
          goto LABEL_56;
      }
      *(_QWORD *)(v16 + 8) = v54[0];
LABEL_17:
      sub_B91220((__int64)v4, v15);
LABEL_18:
      v56 = 257;
      v19 = (char *)sub_BD5D20(v8);
      sub_E0CEC0((__int64)v52, v20, v19);
      v21 = sub_B33830((__int64)v5, v52[0], (signed __int64)v52[1], (__int64)v4, 0, 0, 1);
      if ( (__int64 *)v52[0] != &v53 )
      {
        v47 = v21;
        j_j___libc_free_0((unsigned __int64)v52[0]);
        v21 = v47;
      }
      v54[0] = v21;
      v22 = *(_QWORD *)(v8 + 80);
      if ( !v22 )
        BUG();
      v23 = *(_QWORD *)(v22 + 32);
      if ( v23 )
        v23 -= 24;
      sub_24BF2E0(v8, v23, "__rtsan_notify_blocking_call", v4, 1);
      sub_24BF280((__int64)v4);
      nullsub_61();
      v74 = &unk_49DA100;
      nullsub_63();
      if ( v59 != (unsigned __int64 *)v61 )
        _libc_free((unsigned __int64)v59);
      if ( !v58 )
        _libc_free(v57);
      if ( v55 )
      {
LABEL_3:
        v7 = *(_QWORD *)(v7 + 8);
        if ( v51 == v7 )
          goto LABEL_29;
      }
      else
      {
        _libc_free(v54[1]);
        v7 = *(_QWORD *)(v7 + 8);
        if ( v51 == v7 )
          goto LABEL_29;
      }
    }
    sub_93FB40((__int64)v5, 0);
    v15 = v54[0];
    goto LABEL_60;
  }
LABEL_29:
  memset((void *)a1, 0, 0x60u);
  *(_DWORD *)(a1 + 16) = 2;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_BYTE *)(a1 + 28) = 1;
  *(_DWORD *)(a1 + 64) = 2;
  *(_BYTE *)(a1 + 76) = 1;
  return a1;
}
