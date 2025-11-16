// Function: sub_28AC6E0
// Address: 0x28ac6e0
//
__int64 __fastcall sub_28AC6E0(__int64 a1, __int64 a2)
{
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // rcx
  int v7; // eax
  int v8; // esi
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // rdi
  _BYTE *v12; // r12
  unsigned __int8 *v13; // r14
  unsigned __int8 v14; // al
  unsigned __int8 *v15; // r12
  __int64 v17; // rsi
  unsigned int v18; // eax
  bool v19; // zf
  __int64 v20; // rsi
  __int64 v21; // rax
  _QWORD *v22; // r13
  unsigned __int64 v23; // r13
  unsigned __int8 *v24; // rax
  __int64 v25; // rax
  __int64 *v26; // rax
  __int64 *v27; // rax
  unsigned __int8 **v28; // rax
  _QWORD *v29; // rax
  __int64 v30; // rsi
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rbx
  __int64 v34; // rax
  unsigned __int64 v35; // rax
  unsigned __int8 *v36; // rax
  int v37; // eax
  int v38; // r8d
  __int64 v39; // [rsp+0h] [rbp-3C0h]
  unsigned __int8 *v40; // [rsp+8h] [rbp-3B8h]
  unsigned __int64 v41; // [rsp+8h] [rbp-3B8h]
  _QWORD *v42; // [rsp+10h] [rbp-3B0h] BYREF
  unsigned int v43; // [rsp+18h] [rbp-3A8h]
  __m128i v44; // [rsp+20h] [rbp-3A0h] BYREF
  _QWORD v45[6]; // [rsp+50h] [rbp-370h] BYREF
  _QWORD v46[6]; // [rsp+80h] [rbp-340h] BYREF
  _QWORD v47[6]; // [rsp+B0h] [rbp-310h] BYREF
  __int64 v48; // [rsp+E0h] [rbp-2E0h] BYREF
  _QWORD v49[2]; // [rsp+E8h] [rbp-2D8h] BYREF
  __int64 v50; // [rsp+F8h] [rbp-2C8h]
  __int64 v51; // [rsp+100h] [rbp-2C0h] BYREF
  unsigned int v52; // [rsp+108h] [rbp-2B8h]
  _QWORD v53[2]; // [rsp+240h] [rbp-180h] BYREF
  char v54; // [rsp+250h] [rbp-170h]
  _BYTE *v55; // [rsp+258h] [rbp-168h]
  __int64 v56; // [rsp+260h] [rbp-160h]
  _BYTE v57[128]; // [rsp+268h] [rbp-158h] BYREF
  __int16 v58; // [rsp+2E8h] [rbp-D8h]
  _QWORD v59[2]; // [rsp+2F0h] [rbp-D0h] BYREF
  __int64 v60; // [rsp+300h] [rbp-C0h]
  __int64 v61; // [rsp+308h] [rbp-B8h] BYREF
  unsigned int v62; // [rsp+310h] [rbp-B0h]
  char v63; // [rsp+388h] [rbp-38h] BYREF

  v4 = sub_B43CC0(a2);
  v5 = *(_QWORD *)(a1 + 40);
  v6 = *(_QWORD *)(v5 + 40);
  v7 = *(_DWORD *)(v5 + 56);
  if ( !v7 )
    goto LABEL_6;
  v8 = v7 - 1;
  v9 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v6 + 16LL * v9);
  v11 = *v10;
  if ( a2 != *v10 )
  {
    v37 = 1;
    while ( v11 != -4096 )
    {
      v38 = v37 + 1;
      v9 = v8 & (v37 + v9);
      v10 = (__int64 *)(v6 + 16LL * v9);
      v11 = *v10;
      if ( a2 == *v10 )
        goto LABEL_3;
      v37 = v38;
    }
    goto LABEL_6;
  }
LABEL_3:
  v12 = (_BYTE *)v10[1];
  if ( !v12 )
  {
LABEL_6:
    LODWORD(v15) = 0;
    return (unsigned int)v15;
  }
  sub_D671D0(&v44, a2);
  v13 = sub_BD3990(*(unsigned __int8 **)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))), a2);
  v14 = *v13;
  if ( *v13 <= 0x1Cu )
  {
    if ( v14 != 5 || *((_WORD *)v13 + 1) != 34 )
      goto LABEL_6;
  }
  else if ( v14 != 63 )
  {
    goto LABEL_6;
  }
  v17 = *((_QWORD *)v13 + 1);
  v43 = sub_AE43F0(v4, v17);
  if ( v43 > 0x40 )
  {
    v17 = 0;
    sub_C43690((__int64)&v42, 0, 0);
  }
  else
  {
    v42 = 0;
  }
  v39 = v44.m128i_i64[1];
  v40 = *(unsigned __int8 **)&v13[-32 * (*((_DWORD *)v13 + 1) & 0x7FFFFFF)];
  if ( v40 != sub_BD3990(*(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), v17)
    || v39 == -1
    || v39 == 0xBFFFFFFFFFFFFFFELL )
  {
    v18 = v43;
  }
  else
  {
    v19 = (unsigned __int8)sub_BB6360((__int64)v13, v4, (__int64)&v42, 0, 0) == 0;
    v18 = v43;
    if ( !v19 )
    {
      v20 = (__int64)v42;
      if ( v43 > 0x40 )
        v20 = v42[(v43 - 1) >> 6];
      if ( (v20 & (1LL << ((unsigned __int8)v43 - 1))) == 0 )
      {
        v48 = v39 & 0x3FFFFFFFFFFFFFFFLL;
        LOBYTE(v49[0]) = (v39 & 0x4000000000000000LL) != 0;
        v21 = sub_CA1930(&v48);
        v22 = v42;
        v41 = v21;
        if ( v43 > 0x40 )
          v22 = (_QWORD *)*v42;
        v23 = (unsigned __int64)v22 + v21;
        if ( v23 > 0x3FFFFFFFFFFFFFFBLL )
          v23 = 0xBFFFFFFFFFFFFFFELL;
        v24 = sub_BD3990(*(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), v20);
        v45[1] = v23;
        v45[0] = v24;
        v25 = *(_QWORD *)(a1 + 8);
        memset(&v45[2], 0, 32);
        v49[1] = 0;
        v50 = 1;
        v48 = v25;
        v49[0] = v25;
        v26 = &v51;
        do
        {
          *v26 = -4;
          v26 += 5;
          *(v26 - 4) = -3;
          *(v26 - 3) = -4;
          *(v26 - 2) = -3;
        }
        while ( v26 != v53 );
        v53[1] = 0;
        v53[0] = v59;
        v56 = 0x400000000LL;
        v58 = 256;
        v54 = 0;
        v55 = v57;
        v59[1] = 0;
        v60 = 1;
        v59[0] = &unk_49DDBE8;
        v27 = &v61;
        do
        {
          *v27 = -4096;
          v27 += 2;
        }
        while ( v27 != (__int64 *)&v63 );
        v28 = (unsigned __int8 **)(v12 - 64);
        if ( *v12 == 26 )
          v28 = (unsigned __int8 **)(v12 - 32);
        v15 = *v28;
        v29 = sub_103E0E0(*(_QWORD **)(a1 + 40));
        v30 = (__int64)v15;
        LODWORD(v15) = 0;
        v31 = (*(__int64 (__fastcall **)(_QWORD *, __int64, _QWORD *, __int64 *))(*v29 + 24LL))(v29, v30, v45, &v48);
        if ( *(_BYTE *)v31 == 27 )
        {
          v32 = sub_28AC680(*(_QWORD *)(v31 + 72));
          v33 = v32;
          if ( v32 )
          {
            v34 = *(_QWORD *)(v32 + 32 * (2LL - (*(_DWORD *)(v32 + 4) & 0x7FFFFFF)));
            if ( *(_BYTE *)v34 == 17 )
            {
              v35 = *(_DWORD *)(v34 + 32) <= 0x40u ? *(_QWORD *)(v34 + 24) : **(_QWORD **)(v34 + 24);
              LODWORD(v15) = 0;
              if ( v41 <= v35 )
              {
                v15 = sub_BD3990(*(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), v30);
                v36 = sub_BD3990(*(unsigned __int8 **)(v33 - 32LL * (*(_DWORD *)(v33 + 4) & 0x7FFFFFF)), v30);
                v47[0] = v15;
                v47[1] = 1;
                memset(&v47[2], 0, 32);
                v46[0] = v36;
                v46[1] = 1;
                memset(&v46[2], 0, 32);
                LOBYTE(v15) = (unsigned __int8)sub_CF4D50(v48, (__int64)v46, (__int64)v47, (__int64)v49, 0) == 3;
              }
            }
          }
        }
        v59[0] = &unk_49DDBE8;
        if ( (v60 & 1) == 0 )
          sub_C7D6A0(v61, 16LL * v62, 8);
        nullsub_184();
        if ( v55 != v57 )
          _libc_free((unsigned __int64)v55);
        if ( (v50 & 1) == 0 )
          sub_C7D6A0(v51, 40LL * v52, 8);
        v18 = v43;
        goto LABEL_15;
      }
    }
  }
  LODWORD(v15) = 0;
LABEL_15:
  if ( v18 > 0x40 && v42 )
    j_j___libc_free_0_0((unsigned __int64)v42);
  return (unsigned int)v15;
}
