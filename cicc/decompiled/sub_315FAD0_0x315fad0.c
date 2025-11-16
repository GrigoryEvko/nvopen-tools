// Function: sub_315FAD0
// Address: 0x315fad0
//
unsigned __int64 __fastcall sub_315FAD0(__int64 a1, __int64 a2)
{
  __int64 v4; // rcx
  __int64 v5; // rsi
  unsigned int v6; // edx
  __int64 *v7; // rax
  __int64 v8; // r8
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rax
  bool v14; // zf
  _BYTE **v15; // rcx
  __int64 v16; // r8
  __int64 *v17; // rax
  unsigned int **v18; // rdi
  unsigned __int64 v19; // r14
  int v21; // eax
  __int64 v22; // rax
  unsigned __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // r13
  __int64 v28; // rax
  unsigned __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // rsi
  unsigned int v32; // edx
  __int64 *v33; // rax
  __int64 v34; // r8
  __int64 *v35; // rbx
  __m128i v36; // rax
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // rax
  __int64 **v40; // rax
  __int64 *v41; // rdi
  unsigned __int64 v42; // rcx
  __int64 v43; // rax
  __int64 *v44; // r14
  _BYTE *v45; // rsi
  __int64 v46; // rdi
  __int64 (__fastcall *v47)(__int64, unsigned int, unsigned __int8 *, _BYTE *, unsigned __int8, char); // rax
  __int64 *v48; // r14
  __int64 v49; // r11
  __int64 v50; // rax
  __int64 *v51; // rdi
  __int64 **v52; // rcx
  int v53; // r9d
  __int64 v54; // rdx
  __int64 v55; // r14
  __int64 v56; // rdx
  unsigned int v57; // esi
  __int64 v58; // rax
  __int64 v59; // r14
  __int64 v60; // rdx
  unsigned int v61; // esi
  __int64 v62; // rax
  int v63; // eax
  int v64; // r9d
  __int64 v65; // [rsp+0h] [rbp-120h]
  unsigned __int8 *v66; // [rsp+8h] [rbp-118h]
  __int64 v67; // [rsp+8h] [rbp-118h]
  __int64 v68; // [rsp+8h] [rbp-118h]
  __int64 v69; // [rsp+8h] [rbp-118h]
  __int64 v70; // [rsp+10h] [rbp-110h]
  __int64 v71; // [rsp+10h] [rbp-110h]
  __int64 v72; // [rsp+10h] [rbp-110h]
  __int64 v73; // [rsp+10h] [rbp-110h]
  __int64 v74; // [rsp+10h] [rbp-110h]
  __int64 v75; // [rsp+18h] [rbp-108h]
  __int64 v76; // [rsp+18h] [rbp-108h]
  __int64 v77; // [rsp+18h] [rbp-108h]
  __int64 v78; // [rsp+18h] [rbp-108h]
  int v79; // [rsp+28h] [rbp-F8h]
  _QWORD *v80; // [rsp+30h] [rbp-F0h] BYREF
  __int64 v81; // [rsp+38h] [rbp-E8h]
  _QWORD v82[4]; // [rsp+40h] [rbp-E0h] BYREF
  __m128i v83[2]; // [rsp+60h] [rbp-C0h] BYREF
  __int16 v84; // [rsp+80h] [rbp-A0h]
  __m128i v85[2]; // [rsp+90h] [rbp-90h] BYREF
  __int16 v86; // [rsp+B0h] [rbp-70h]
  __m128i v87[2]; // [rsp+C0h] [rbp-60h] BYREF
  __int16 v88; // [rsp+E0h] [rbp-40h]

  v4 = *(unsigned int *)(*(_QWORD *)a1 + 48LL);
  v5 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
  if ( (_DWORD)v4 )
  {
    v6 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v5 + 16LL * v6);
    v8 = *v7;
    if ( a2 == *v7 )
      goto LABEL_3;
    v21 = 1;
    while ( v8 != -4096 )
    {
      v53 = v21 + 1;
      v6 = (v4 - 1) & (v21 + v6);
      v7 = (__int64 *)(v5 + 16LL * v6);
      v8 = *v7;
      if ( a2 == *v7 )
        goto LABEL_3;
      v21 = v53;
    }
  }
  v7 = (__int64 *)(v5 + 16 * v4);
LABEL_3:
  v9 = *((unsigned int *)v7 + 2);
  v10 = sub_BCB2D0(*(_QWORD **)(a1 + 8));
  v11 = sub_ACD640(v10, 0, 0);
  v12 = sub_BCB2D0(*(_QWORD **)(a1 + 8));
  v13 = sub_ACD640(v12, v9, 0);
  v14 = *(_BYTE *)a2 == 60;
  v15 = (_BYTE **)v82;
  v80 = v82;
  v82[1] = v13;
  v16 = 2;
  v82[0] = v11;
  v81 = 0x300000002LL;
  if ( v14 )
  {
    v22 = *(_QWORD *)(a2 - 32);
    if ( *(_BYTE *)v22 != 17 )
      sub_C64ED0("Coroutines cannot handle non static allocas yet", 1u);
    if ( *(_DWORD *)(v22 + 32) <= 0x40u )
      v23 = *(_QWORD *)(v22 + 24);
    else
      v23 = **(_QWORD **)(v22 + 24);
    v16 = 2;
    v15 = (_BYTE **)v82;
    if ( v23 > 1 )
    {
      v24 = sub_BCB2D0(*(_QWORD **)(a1 + 8));
      v27 = sub_ACD640(v24, 0, 0);
      v28 = (unsigned int)v81;
      v29 = (unsigned int)v81 + 1LL;
      if ( v29 > HIDWORD(v81) )
      {
        sub_C8D5F0((__int64)&v80, v82, v29, 8u, v25, v26);
        v28 = (unsigned int)v81;
      }
      v80[v28] = v27;
      v15 = (_BYTE **)v80;
      v16 = (unsigned int)(v81 + 1);
      LODWORD(v81) = v81 + 1;
    }
  }
  v17 = *(__int64 **)(a1 + 32);
  v18 = *(unsigned int ***)(a1 + 16);
  v88 = 257;
  v19 = sub_921130(v18, **(_QWORD **)(a1 + 24), *v17, v15, v16, (__int64)v87, 3u);
  if ( *(_BYTE *)a2 == 60 )
  {
    v30 = *(unsigned int *)(*(_QWORD *)a1 + 112LL);
    v31 = *(_QWORD *)(*(_QWORD *)a1 + 96LL);
    if ( (_DWORD)v30 )
    {
      v32 = (v30 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v33 = (__int64 *)(v31 + 16LL * v32);
      v34 = *v33;
      if ( a2 == *v33 )
        goto LABEL_20;
      v63 = 1;
      while ( v34 != -4096 )
      {
        v64 = v63 + 1;
        v32 = (v30 - 1) & (v63 + v32);
        v33 = (__int64 *)(v31 + 16LL * v32);
        v34 = *v33;
        if ( a2 == *v33 )
          goto LABEL_20;
        v63 = v64;
      }
    }
    v33 = (__int64 *)(v31 + 16 * v30);
LABEL_20:
    if ( !v33[1] )
    {
      if ( *(_QWORD *)(a2 + 8) != *(_QWORD *)(v19 + 8) )
      {
        v35 = *(__int64 **)(a1 + 16);
        v83[0].m128i_i64[0] = (__int64)".cast";
        v84 = 259;
        v36.m128i_i64[0] = (__int64)sub_BD5D20(a2);
        v86 = 261;
        v85[0] = v36;
        sub_9C6370(v87, v85, v83, 261, v37, v38);
        v19 = sub_315F700(v35, 0x32u, v19, *(__int64 ***)(a2 + 8), (__int64)v87, 0, v79, 0);
      }
      goto LABEL_5;
    }
    v39 = sub_B43CA0(a2);
    v40 = (__int64 **)sub_AE4450(v39 + 312, *(_QWORD *)(a2 + 8));
    v41 = *(__int64 **)(a1 + 16);
    v88 = 257;
    v75 = (__int64)v40;
    v66 = (unsigned __int8 *)sub_315F700(v41, 0x2Fu, v19, v40, (__int64)v87, 0, v85[0].m128i_i32[0], 0);
    _BitScanReverse64(&v42, 1LL << *(_WORD *)(a2 + 2));
    v43 = sub_AD64C0(v75, (0x8000000000000000LL >> ((unsigned __int8)v42 ^ 0x3Fu)) - 1, 0);
    v44 = *(__int64 **)(a1 + 16);
    v86 = 257;
    v45 = (_BYTE *)v43;
    v46 = v44[10];
    v70 = v43;
    v47 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, _BYTE *, unsigned __int8, char))(*(_QWORD *)v46 + 32LL);
    if ( v47 == sub_9201A0 )
    {
      if ( *v66 > 0x15u || *v45 > 0x15u )
        goto LABEL_38;
      if ( (unsigned __int8)sub_AC47B0(13) )
        v76 = sub_AD5570(13, (__int64)v66, (unsigned __int8 *)v70, 0, 0);
      else
        v76 = sub_AABE40(0xDu, v66, (unsigned __int8 *)v70);
    }
    else
    {
      v76 = v47(v46, 13u, v66, (_BYTE *)v70, 0, 0);
    }
    if ( v76 )
    {
LABEL_30:
      v48 = *(__int64 **)(a1 + 16);
      v86 = 257;
      v84 = 257;
      v67 = sub_AD62B0(*(_QWORD *)(v70 + 8));
      v49 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v48[10] + 16LL))(
              v48[10],
              30,
              v70,
              v67);
      if ( !v49 )
      {
        v88 = 257;
        v73 = sub_B504D0(30, v70, v67, (__int64)v87, 0, 0);
        (*(void (__fastcall **)(__int64, __int64, __m128i *, __int64, __int64))(*(_QWORD *)v48[11] + 16LL))(
          v48[11],
          v73,
          v83,
          v48[7],
          v48[8]);
        v62 = *v48;
        v49 = v73;
        v65 = *v48 + 16LL * *((unsigned int *)v48 + 2);
        if ( *v48 != v65 )
        {
          do
          {
            v69 = v62;
            v74 = v49;
            sub_B99FD0(v49, *(_DWORD *)v62, *(_QWORD *)(v62 + 8));
            v49 = v74;
            v62 = v69 + 16;
          }
          while ( v65 != v69 + 16 );
        }
      }
      v71 = v49;
      v50 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v48[10] + 16LL))(
              v48[10],
              28,
              v76,
              v49);
      if ( !v50 )
      {
        v88 = 257;
        v77 = sub_B504D0(28, v76, v71, (__int64)v87, 0, 0);
        (*(void (__fastcall **)(__int64, __int64, __m128i *, __int64, __int64))(*(_QWORD *)v48[11] + 16LL))(
          v48[11],
          v77,
          v85,
          v48[7],
          v48[8]);
        v58 = *v48 + 16LL * *((unsigned int *)v48 + 2);
        v59 = *v48;
        v14 = v59 == v58;
        v72 = v58;
        v50 = v77;
        if ( !v14 )
        {
          do
          {
            v60 = *(_QWORD *)(v59 + 8);
            v61 = *(_DWORD *)v59;
            v59 += 16;
            v78 = v50;
            sub_B99FD0(v50, v61, v60);
            v50 = v78;
          }
          while ( v72 != v59 );
        }
      }
      v51 = *(__int64 **)(a1 + 16);
      v52 = *(__int64 ***)(a2 + 8);
      v88 = 257;
      v19 = sub_315F700(v51, 0x30u, v50, v52, (__int64)v87, 0, v85[0].m128i_i32[0], 0);
      goto LABEL_5;
    }
LABEL_38:
    v88 = 257;
    v76 = sub_B504D0(13, (__int64)v66, v70, (__int64)v87, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, __m128i *, __int64, __int64))(*(_QWORD *)v44[11] + 16LL))(
      v44[11],
      v76,
      v85,
      v44[7],
      v44[8]);
    v54 = 16LL * *((unsigned int *)v44 + 2);
    v55 = *v44;
    v68 = v55 + v54;
    while ( v68 != v55 )
    {
      v56 = *(_QWORD *)(v55 + 8);
      v57 = *(_DWORD *)v55;
      v55 += 16;
      sub_B99FD0(v76, v57, v56);
    }
    goto LABEL_30;
  }
LABEL_5:
  if ( v80 != v82 )
    _libc_free((unsigned __int64)v80);
  return v19;
}
