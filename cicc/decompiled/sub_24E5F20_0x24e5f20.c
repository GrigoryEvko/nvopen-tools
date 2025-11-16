// Function: sub_24E5F20
// Address: 0x24e5f20
//
__int64 __fastcall sub_24E5F20(__int64 *a1, __int64 a2, __int64 a3, unsigned __int64 *a4, __int64 a5, __int64 a6)
{
  unsigned int **v6; // r13
  unsigned __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 *v10; // rbx
  __int64 v12; // rax
  int v13; // ecx
  char v14; // dl
  int v15; // edx
  __int64 v16; // rdi
  __int64 (__fastcall *v17)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v18; // r13
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // r14
  __int64 v22; // r15
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  int v25; // edi
  unsigned __int64 v26; // rax
  __int64 v27; // rdx
  int v28; // r14d
  __int64 v29; // r14
  __int64 v30; // r15
  __int64 v31; // rdx
  unsigned int v32; // esi
  int v33; // ecx
  int v34; // r8d
  __int64 v35; // r12
  __int64 *v36; // r13
  __int64 v37; // rsi
  __int64 v39; // rsi
  unsigned __int8 *v40; // rsi
  __int64 v41; // [rsp-10h] [rbp-140h]
  unsigned __int64 v44; // [rsp+20h] [rbp-110h]
  __int64 v46; // [rsp+40h] [rbp-F0h]
  __int64 *v47; // [rsp+48h] [rbp-E8h]
  _BYTE v48[32]; // [rsp+50h] [rbp-E0h] BYREF
  __int16 v49; // [rsp+70h] [rbp-C0h]
  __int64 v50[4]; // [rsp+80h] [rbp-B0h] BYREF
  __int16 v51; // [rsp+A0h] [rbp-90h]
  _BYTE *v52; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v53; // [rsp+B8h] [rbp-78h]
  _BYTE v54[112]; // [rsp+C0h] [rbp-70h] BYREF

  v6 = (unsigned int **)a6;
  v7 = *(_QWORD *)(a2 + 24);
  v52 = v54;
  v53 = 0x800000000LL;
  v8 = *(unsigned int *)(v7 + 12);
  v44 = v7;
  v9 = *(_QWORD *)(v7 + 16);
  v10 = (__int64 *)(v9 + 8);
  v47 = (__int64 *)(v9 + 8 * v8);
  if ( (__int64 *)(v9 + 8) != v47 )
  {
    v46 = a6;
    while ( 1 )
    {
      v21 = *a4;
      v22 = *v10;
      if ( *v10 == *(_QWORD *)(*a4 + 8) )
      {
        v23 = (unsigned int)v53;
        v24 = (unsigned int)v53 + 1LL;
        if ( v24 > HIDWORD(v53) )
        {
          sub_C8D5F0((__int64)&v52, v54, v24, 8u, a5, a6);
          v23 = (unsigned int)v53;
        }
        *(_QWORD *)&v52[8 * v23] = v21;
        LODWORD(v53) = v53 + 1;
        goto LABEL_19;
      }
      v49 = 257;
      v12 = *(_QWORD *)(v21 + 8);
      if ( v22 == v12 )
        break;
      v13 = *(unsigned __int8 *)(v12 + 8);
      v14 = *(_BYTE *)(v12 + 8);
      if ( (unsigned int)(v13 - 17) > 1 )
      {
        if ( (_BYTE)v13 != 14 )
          goto LABEL_25;
      }
      else if ( *(_BYTE *)(**(_QWORD **)(v12 + 16) + 8LL) != 14 )
      {
        goto LABEL_6;
      }
      v25 = *(unsigned __int8 *)(v22 + 8);
      if ( (unsigned int)(v25 - 17) <= 1 )
        LOBYTE(v25) = *(_BYTE *)(**(_QWORD **)(v22 + 16) + 8LL);
      if ( (_BYTE)v25 != 12 )
      {
LABEL_6:
        if ( v13 == 18 )
          goto LABEL_7;
LABEL_25:
        if ( v13 == 17 )
LABEL_7:
          v14 = *(_BYTE *)(**(_QWORD **)(v12 + 16) + 8LL);
        if ( v14 != 12 )
          goto LABEL_12;
        v15 = *(unsigned __int8 *)(v22 + 8);
        if ( (unsigned int)(v15 - 17) <= 1 )
          LOBYTE(v15) = *(_BYTE *)(**(_QWORD **)(v22 + 16) + 8LL);
        if ( (_BYTE)v15 == 14 )
        {
          v18 = sub_24E55A0((__int64 *)v46, 0x30u, v21, (__int64 **)v22, (__int64)v48, 0, v50[0], 0);
        }
        else
        {
LABEL_12:
          v16 = *(_QWORD *)(v46 + 80);
          v17 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v16 + 120LL);
          if ( v17 != sub_920130 )
          {
            v18 = v17(v16, 49u, (_BYTE *)v21, v22);
            goto LABEL_16;
          }
          if ( *(_BYTE *)v21 > 0x15u )
            goto LABEL_33;
          v18 = (unsigned __int8)sub_AC4810(0x31u)
              ? sub_ADAB70(49, v21, (__int64 **)v22, 0)
              : sub_AA93C0(0x31u, v21, v22);
LABEL_16:
          if ( !v18 )
          {
LABEL_33:
            v51 = 257;
            v18 = sub_B51D30(49, v21, v22, (__int64)v50, 0, 0);
            if ( (unsigned __int8)sub_920620(v18) )
            {
              v27 = *(_QWORD *)(v46 + 96);
              v28 = *(_DWORD *)(v46 + 104);
              if ( v27 )
                sub_B99FD0(v18, 3u, v27);
              sub_B45150(v18, v28);
            }
            (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v46 + 88) + 16LL))(
              *(_QWORD *)(v46 + 88),
              v18,
              v48,
              *(_QWORD *)(v46 + 56),
              *(_QWORD *)(v46 + 64));
            v29 = *(_QWORD *)v46 + 16LL * *(unsigned int *)(v46 + 8);
            if ( *(_QWORD *)v46 != v29 )
            {
              v30 = *(_QWORD *)v46;
              do
              {
                v31 = *(_QWORD *)(v30 + 8);
                v32 = *(_DWORD *)v30;
                v30 += 16;
                sub_B99FD0(v18, v32, v31);
              }
              while ( v29 != v30 );
            }
          }
        }
LABEL_17:
        v19 = (unsigned int)v53;
        v20 = (unsigned int)v53 + 1LL;
        if ( v20 > HIDWORD(v53) )
          goto LABEL_31;
        goto LABEL_18;
      }
      v26 = sub_24E55A0((__int64 *)v46, 0x2Fu, v21, (__int64 **)v22, (__int64)v48, 0, v50[0], 0);
      a6 = v41;
      v18 = v26;
      v19 = (unsigned int)v53;
      v20 = (unsigned int)v53 + 1LL;
      if ( v20 > HIDWORD(v53) )
      {
LABEL_31:
        sub_C8D5F0((__int64)&v52, v54, v20, 8u, a5, a6);
        v19 = (unsigned int)v53;
      }
LABEL_18:
      *(_QWORD *)&v52[8 * v19] = v18;
      LODWORD(v53) = v53 + 1;
LABEL_19:
      ++v10;
      ++a4;
      if ( v47 == v10 )
      {
        v6 = (unsigned int **)v46;
        v33 = (int)v52;
        v34 = v53;
        goto LABEL_43;
      }
    }
    v18 = v21;
    goto LABEL_17;
  }
  v33 = (unsigned int)v54;
  v34 = 0;
LABEL_43:
  v51 = 257;
  v35 = sub_921880(v6, v44, a2, v33, v34, (__int64)v50, 0);
  if ( (unsigned __int8)sub_DFAB60(a3) )
    *(_WORD *)(v35 + 2) = *(_WORD *)(v35 + 2) & 0xFFFC | 2;
  v36 = (__int64 *)(v35 + 48);
  v37 = *a1;
  v50[0] = v37;
  if ( !v37 )
  {
    if ( v36 == v50 )
      goto LABEL_49;
    v39 = *(_QWORD *)(v35 + 48);
    if ( !v39 )
      goto LABEL_49;
LABEL_54:
    sub_B91220(v35 + 48, v39);
    goto LABEL_55;
  }
  sub_B96E90((__int64)v50, v37, 1);
  if ( v36 == v50 )
  {
    if ( v50[0] )
      sub_B91220((__int64)v50, v50[0]);
    goto LABEL_49;
  }
  v39 = *(_QWORD *)(v35 + 48);
  if ( v39 )
    goto LABEL_54;
LABEL_55:
  v40 = (unsigned __int8 *)v50[0];
  *(_QWORD *)(v35 + 48) = v50[0];
  if ( v40 )
    sub_B976B0((__int64)v50, v40, v35 + 48);
LABEL_49:
  *(_WORD *)(v35 + 2) = *(_WORD *)(v35 + 2) & 0xF003 | (4 * ((*(_WORD *)(a2 + 2) >> 4) & 0x3FF));
  if ( v52 != v54 )
    _libc_free((unsigned __int64)v52);
  return v35;
}
