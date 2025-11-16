// Function: sub_2439370
// Address: 0x2439370
//
void __fastcall sub_2439370(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  int v4; // eax
  int v5; // edx
  __int64 **v6; // r15
  __int64 v7; // rdx
  unsigned int v8; // esi
  unsigned __int64 v9; // rax
  bool v10; // zf
  __int64 *v11; // rcx
  int v12; // r8d
  __int64 v13; // rax
  unsigned __int64 *v14; // rdx
  unsigned __int64 v15; // rsi
  unsigned __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  int v19; // r8d
  __int64 *v20; // rdx
  int v21; // eax
  __int64 v22; // rax
  __int64 v23; // rax
  int v24; // ecx
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdx
  int v29; // r8d
  __int64 *v30; // rdx
  unsigned __int64 v31; // [rsp+0h] [rbp-180h]
  unsigned int v32; // [rsp+8h] [rbp-178h]
  __int64 v33; // [rsp+10h] [rbp-170h]
  __int64 v34; // [rsp+10h] [rbp-170h]
  __int64 v35; // [rsp+18h] [rbp-168h]
  __int64 v36; // [rsp+18h] [rbp-168h]
  __int64 v37; // [rsp+18h] [rbp-168h]
  __int64 v38; // [rsp+18h] [rbp-168h]
  char v39[32]; // [rsp+30h] [rbp-150h] BYREF
  __int16 v40; // [rsp+50h] [rbp-130h]
  _BYTE v41[32]; // [rsp+60h] [rbp-120h] BYREF
  __int16 v42; // [rsp+80h] [rbp-100h]
  __int64 *v43; // [rsp+90h] [rbp-F0h] BYREF
  __int64 v44; // [rsp+98h] [rbp-E8h]
  __int64 v45; // [rsp+A0h] [rbp-E0h] BYREF
  __int64 v46; // [rsp+A8h] [rbp-D8h]
  unsigned __int64 v47; // [rsp+B0h] [rbp-D0h]
  unsigned int *v48[2]; // [rsp+C0h] [rbp-C0h] BYREF
  _BYTE v49[32]; // [rsp+D0h] [rbp-B0h] BYREF
  __int64 v50; // [rsp+F0h] [rbp-90h]
  __int64 v51; // [rsp+F8h] [rbp-88h]
  __int16 v52; // [rsp+100h] [rbp-80h]
  _QWORD *v53; // [rsp+108h] [rbp-78h]
  void **v54; // [rsp+110h] [rbp-70h]
  void **v55; // [rsp+118h] [rbp-68h]
  __int64 v56; // [rsp+120h] [rbp-60h]
  int v57; // [rsp+128h] [rbp-58h]
  __int16 v58; // [rsp+12Ch] [rbp-54h]
  char v59; // [rsp+12Eh] [rbp-52h]
  __int64 v60; // [rsp+130h] [rbp-50h]
  __int64 v61; // [rsp+138h] [rbp-48h]
  void *v62; // [rsp+140h] [rbp-40h] BYREF
  void *v63; // [rsp+148h] [rbp-38h] BYREF

  v53 = (_QWORD *)sub_BD5C60(a2);
  v55 = &v63;
  v58 = 512;
  v52 = 0;
  v48[1] = (unsigned int *)0x200000000LL;
  v62 = &unk_49DA100;
  v48[0] = (unsigned int *)v49;
  v54 = &v62;
  v63 = &unk_49DA0B0;
  v56 = 0;
  v57 = 0;
  v59 = 7;
  v60 = 0;
  v61 = 0;
  v50 = 0;
  v51 = 0;
  sub_D5F1F0((__int64)v48, a2);
  v3 = *(_QWORD *)(a2 - 32);
  if ( !v3 || *(_BYTE *)v3 )
    BUG();
  if ( *(_QWORD *)(v3 + 24) != *(_QWORD *)(a2 + 80) )
    goto LABEL_36;
  v4 = *(_DWORD *)(v3 + 36);
  if ( v4 == 238 || (unsigned int)(v4 - 240) <= 1 )
  {
    v5 = *(_DWORD *)(a2 + 4);
    v6 = *(__int64 ***)(a1 + 120);
    v42 = 257;
    v7 = v5 & 0x7FFFFFF;
    v33 = *(_QWORD *)(a2 - 32 * v7);
    v35 = *(_QWORD *)(a2 + 32 * (1 - v7));
    v31 = *(_QWORD *)(a2 + 32 * (2 - v7));
    v32 = sub_BCB060(*(_QWORD *)(v31 + 8));
    v8 = (v32 <= (unsigned int)sub_BCB060((__int64)v6)) + 38;
    v9 = sub_2436E50((__int64 *)v48, v8, v31, v6, (__int64)v41, 0, (int)v43, 0);
    v43 = &v45;
    v47 = v9;
    v10 = *(_BYTE *)(a1 + 171) == 0;
    v46 = v35;
    v11 = &v45;
    v12 = 3;
    v44 = 0x400000003LL;
    v45 = v33;
    if ( !v10 )
    {
      v17 = sub_AD64C0(*(_QWORD *)(a1 + 136), *(unsigned __int8 *)(a1 + 172), 0);
      v18 = (unsigned int)v44;
      v19 = v44;
      if ( (unsigned int)v44 >= (unsigned __int64)HIDWORD(v44) )
      {
        if ( HIDWORD(v44) < (unsigned __int64)(unsigned int)v44 + 1 )
        {
          v37 = v17;
          sub_C8D5F0((__int64)&v43, &v45, (unsigned int)v44 + 1LL, 8u, (unsigned int)v44 + 1LL, (__int64)&v43);
          v18 = (unsigned int)v44;
          v17 = v37;
        }
        v43[v18] = v17;
        LODWORD(v11) = (_DWORD)v43;
        v12 = v44 + 1;
        LODWORD(v44) = v44 + 1;
      }
      else
      {
        LODWORD(v11) = (_DWORD)v43;
        v20 = &v43[(unsigned int)v44];
        if ( v20 )
        {
          *v20 = v17;
          v19 = v44;
          LODWORD(v11) = (_DWORD)v43;
        }
        v12 = v19 + 1;
        LODWORD(v44) = v12;
      }
    }
    v13 = *(_QWORD *)(a2 - 32);
    v42 = 257;
    if ( v13 && !*(_BYTE *)v13 && *(_QWORD *)(v13 + 24) == *(_QWORD *)(a2 + 80) )
    {
      v14 = (unsigned __int64 *)(a1 + 408);
      if ( *(_DWORD *)(v13 + 36) == 241 )
        v14 = (unsigned __int64 *)(a1 + 392);
      v15 = *v14;
      v16 = v14[1];
      goto LABEL_13;
    }
LABEL_36:
    BUG();
  }
  if ( ((v4 - 243) & 0xFFFFFFFD) != 0 )
    goto LABEL_15;
  v21 = *(_DWORD *)(a2 + 4);
  v40 = 257;
  v34 = *(_QWORD *)(a2 - 32LL * (v21 & 0x7FFFFFF));
  v22 = sub_BCB2D0(v53);
  v23 = sub_921630(v48, *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))), v22, 0, (__int64)v39);
  v24 = *(_DWORD *)(a2 + 4);
  v25 = *(_QWORD *)(a1 + 120);
  v36 = v23;
  v42 = 257;
  v26 = sub_921630(v48, *(_QWORD *)(a2 + 32 * (2LL - (v24 & 0x7FFFFFF))), v25, 0, (__int64)v41);
  v11 = &v45;
  v47 = v26;
  v10 = *(_BYTE *)(a1 + 171) == 0;
  v43 = &v45;
  v12 = 3;
  v45 = v34;
  v46 = v36;
  v44 = 0x400000003LL;
  if ( !v10 )
  {
    v27 = sub_AD64C0(*(_QWORD *)(a1 + 136), *(unsigned __int8 *)(a1 + 172), 0);
    v28 = (unsigned int)v44;
    v29 = v44;
    if ( (unsigned int)v44 >= (unsigned __int64)HIDWORD(v44) )
    {
      if ( HIDWORD(v44) < (unsigned __int64)(unsigned int)v44 + 1 )
      {
        v38 = v27;
        sub_C8D5F0((__int64)&v43, &v45, (unsigned int)v44 + 1LL, 8u, (unsigned int)v44 + 1LL, (__int64)&v43);
        v28 = (unsigned int)v44;
        v27 = v38;
      }
      v43[v28] = v27;
      LODWORD(v11) = (_DWORD)v43;
      v12 = v44 + 1;
      LODWORD(v44) = v44 + 1;
    }
    else
    {
      LODWORD(v11) = (_DWORD)v43;
      v30 = &v43[(unsigned int)v44];
      if ( v30 )
      {
        *v30 = v27;
        v29 = v44;
        LODWORD(v11) = (_DWORD)v43;
      }
      v12 = v29 + 1;
      LODWORD(v44) = v12;
    }
  }
  v16 = *(_QWORD *)(a1 + 432);
  v42 = 257;
  v15 = *(_QWORD *)(a1 + 424);
LABEL_13:
  sub_921880(v48, v15, v16, (int)v11, v12, (__int64)v41, 0);
  if ( v43 != &v45 )
    _libc_free((unsigned __int64)v43);
LABEL_15:
  sub_B43D60((_QWORD *)a2);
  nullsub_61();
  v62 = &unk_49DA100;
  nullsub_63();
  if ( (_BYTE *)v48[0] != v49 )
    _libc_free((unsigned __int64)v48[0]);
}
