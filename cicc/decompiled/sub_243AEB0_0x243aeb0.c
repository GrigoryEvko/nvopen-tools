// Function: sub_243AEB0
// Address: 0x243aeb0
//
void __fastcall sub_243AEB0(__int64 a1, __int64 a2, char a3)
{
  __int64 v5; // rbx
  int v6; // eax
  _BYTE *v7; // r13
  __int64 **v8; // r13
  _BYTE *v9; // rax
  __int64 v10; // rdi
  __int64 v11; // r15
  __int64 v12; // rax
  _QWORD *v13; // rax
  unsigned int *v14; // rbx
  __int64 v15; // r15
  __int64 v16; // rdx
  unsigned int v17; // esi
  __int64 v18; // rdi
  __int64 v19; // r13
  __int64 v20; // rax
  char v21; // al
  char v22; // r14
  unsigned int *v23; // r14
  __int64 v24; // r13
  __int64 v25; // rdx
  unsigned int v26; // esi
  __int64 v27; // rax
  __int64 v28; // rdi
  unsigned __int8 *v29; // r14
  __int64 (__fastcall *v30)(__int64, unsigned int, _QWORD *, unsigned __int8 *, unsigned __int8); // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 *v33; // rdi
  __int64 v34; // r14
  __int64 **v35; // rax
  __int64 v36; // rax
  char v37; // al
  _QWORD *v38; // rax
  __int64 v39; // r9
  __int64 v40; // r15
  __int64 v41; // r14
  unsigned int *v42; // rbx
  __int64 v43; // rdx
  unsigned int v44; // esi
  __int64 v45; // r14
  __int64 v46; // rax
  char v47; // al
  _QWORD *v48; // rax
  __int64 v49; // r9
  __int64 v50; // r15
  __int64 v51; // r14
  unsigned int *v52; // rbx
  __int64 v53; // rdx
  unsigned int v54; // esi
  __int64 v55; // rax
  unsigned __int64 v56; // rsi
  __int64 v57; // rdx
  __int64 v58; // r15
  __int64 v59; // rdi
  __int64 v60; // rax
  __int64 v61; // rdi
  _BYTE *v62; // r14
  _BYTE *v63; // rax
  _BYTE *v64; // rax
  unsigned __int64 v65; // rax
  unsigned __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // rdi
  __int64 v71; // r14
  __int64 v72; // rax
  char v73; // al
  _QWORD *v74; // rax
  __int64 v75; // r9
  unsigned int *v76; // r15
  __int64 v77; // r14
  __int64 v78; // rdx
  unsigned int v79; // esi
  __int64 v80; // [rsp-8h] [rbp-D8h]
  __int64 v81; // [rsp+8h] [rbp-C8h]
  unsigned __int64 v82; // [rsp+10h] [rbp-C0h]
  char v83; // [rsp+1Ch] [rbp-B4h]
  char v84; // [rsp+1Ch] [rbp-B4h]
  __int64 v85; // [rsp+20h] [rbp-B0h]
  __int64 v86; // [rsp+20h] [rbp-B0h]
  char v87; // [rsp+20h] [rbp-B0h]
  __int64 v88; // [rsp+28h] [rbp-A8h]
  char v89; // [rsp+30h] [rbp-A0h]
  _QWORD *v90; // [rsp+30h] [rbp-A0h]
  __int64 v91; // [rsp+38h] [rbp-98h]
  int v92[8]; // [rsp+40h] [rbp-90h] BYREF
  __int16 v93; // [rsp+60h] [rbp-70h]
  _QWORD v94[4]; // [rsp+70h] [rbp-60h] BYREF
  __int16 v95; // [rsp+90h] [rbp-40h]

  v5 = a1;
  v6 = *(_DWORD *)(a1 + 88);
  if ( v6 == 3 )
  {
    if ( !a3 )
    {
      if ( *(_DWORD *)(a1 + 72) == 17 )
      {
        v68 = sub_2437EA0(*(__int64 **)(a1 + 128), (unsigned int **)a2, *(_QWORD *)(a1 + 504));
        *(_QWORD *)(a1 + 512) = v68;
        v7 = (_BYTE *)v68;
      }
      else
      {
        v7 = *(_BYTE **)(a1 + 512);
      }
      goto LABEL_5;
    }
  }
  else
  {
    v8 = *(__int64 ***)(a1 + 128);
    if ( v6 )
    {
      if ( v6 == 2 )
      {
        v7 = (_BYTE *)sub_2437EA0(*(__int64 **)(a1 + 128), (unsigned int **)a2, *(_QWORD *)(a1 + 504));
        *(_QWORD *)(a1 + 512) = v7;
        if ( !a3 )
          goto LABEL_5;
      }
      else
      {
        v9 = sub_BA8D60(
               *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 48) + 72LL) + 40LL),
               (__int64)"__hwasan_shadow_memory_dynamic_address",
               0x26u,
               *(_QWORD *)(a1 + 128));
        v10 = *(_QWORD *)(a2 + 48);
        v11 = *(_QWORD *)(v5 + 128);
        v93 = 257;
        v88 = (__int64)v9;
        v12 = sub_AA4E30(v10);
        v95 = 257;
        v89 = sub_AE5020(v12, v11);
        v13 = sub_BD2C40(80, unk_3F10A14);
        v7 = v13;
        if ( v13 )
          sub_B4D190((__int64)v13, v11, v88, (__int64)v94, 0, v89, 0, 0);
        (*(void (__fastcall **)(_QWORD, _BYTE *, int *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
          *(_QWORD *)(a2 + 88),
          v7,
          v92,
          *(_QWORD *)(a2 + 56),
          *(_QWORD *)(a2 + 64));
        if ( *(_QWORD *)a2 != *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8) )
        {
          v91 = v5;
          v14 = *(unsigned int **)a2;
          v15 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
          do
          {
            v16 = *((_QWORD *)v14 + 1);
            v17 = *v14;
            v14 += 4;
            sub_B99FD0((__int64)v7, v17, v16);
          }
          while ( (unsigned int *)v15 != v14 );
          v5 = v91;
        }
        *(_QWORD *)(v5 + 512) = v7;
        if ( !a3 )
        {
LABEL_5:
          if ( v7 )
            return;
          goto LABEL_54;
        }
      }
    }
    else
    {
      v66 = sub_AD64C0(*(_QWORD *)(a1 + 120), *(_QWORD *)(a1 + 96), 0);
      v67 = sub_AD4C70(v66, v8, 0);
      v7 = (_BYTE *)sub_2437EA0(*(__int64 **)(a1 + 128), (unsigned int **)a2, v67);
      *(_QWORD *)(a1 + 512) = v7;
      if ( !a3 )
        goto LABEL_5;
    }
  }
  if ( dword_4FE4348 == 2 )
  {
    v55 = sub_2434E10(v5, (__int64 *)a2);
    v56 = *(_QWORD *)(v5 + 488);
    v95 = 257;
    v57 = *(_QWORD *)(v5 + 496);
    *(_QWORD *)v92 = v55;
    sub_921880((unsigned int **)a2, v56, v57, (int)v92, 1, (__int64)v94, 0);
    if ( *(_QWORD *)(v5 + 512) )
      return;
    goto LABEL_53;
  }
  if ( (unsigned int)dword_4FE4348 > 2 )
  {
    if ( *(_QWORD *)(v5 + 512) )
      return;
LABEL_53:
    v7 = 0;
    goto LABEL_54;
  }
  if ( !dword_4FE4348 )
    BUG();
  if ( (unsigned int)(*(_DWORD *)(v5 + 56) - 3) <= 2 && *(_DWORD *)(v5 + 72) == 17 )
    v81 = sub_2A3A9B0(a2, 6);
  else
    v81 = *(_QWORD *)(v5 + 536);
  v18 = *(_QWORD *)(a2 + 48);
  v19 = *(_QWORD *)(v5 + 120);
  v93 = 257;
  v20 = sub_AA4E30(v18);
  v21 = sub_AE5020(v20, v19);
  v95 = 257;
  v22 = v21;
  v90 = sub_BD2C40(80, unk_3F10A14);
  if ( v90 )
    sub_B4D190((__int64)v90, v19, v81, (__int64)v94, 0, v22, 0, 0);
  (*(void (__fastcall **)(_QWORD, _QWORD *, int *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
    *(_QWORD *)(a2 + 88),
    v90,
    v92,
    *(_QWORD *)(a2 + 56),
    *(_QWORD *)(a2 + 64));
  v23 = *(unsigned int **)a2;
  v24 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 != v24 )
  {
    do
    {
      v25 = *((_QWORD *)v23 + 1);
      v26 = *v23;
      v23 += 4;
      sub_B99FD0((__int64)v90, v26, v25);
    }
    while ( (unsigned int *)v24 != v23 );
  }
  v7 = v90;
  if ( (unsigned int)(*(_DWORD *)(v5 + 56) - 3) > 2 )
    v7 = (_BYTE *)sub_2435400(
                    *(_BYTE *)(v5 + 160),
                    *(_DWORD *)(v5 + 176),
                    *(_QWORD *)(v5 + 184),
                    (__int64 *)a2,
                    (__int64)v90);
  v93 = 257;
  v27 = sub_AD64C0(v90[1], 3, 0);
  v28 = *(_QWORD *)(a2 + 80);
  v29 = (unsigned __int8 *)v27;
  v30 = *(__int64 (__fastcall **)(__int64, unsigned int, _QWORD *, unsigned __int8 *, unsigned __int8))(*(_QWORD *)v28 + 24LL);
  if ( (char *)v30 != (char *)sub_920250 )
  {
    v31 = v30(v28, 27u, v90, v29, 0);
    goto LABEL_33;
  }
  if ( *(_BYTE *)v90 <= 0x15u && *v29 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(27) )
      v31 = sub_AD5570(27, (__int64)v90, v29, 0, 0);
    else
      v31 = sub_AABE40(0x1Bu, (unsigned __int8 *)v90, v29);
LABEL_33:
    if ( v31 )
      goto LABEL_34;
  }
  v95 = 257;
  v69 = sub_B504D0(27, (__int64)v90, (__int64)v29, (__int64)v94, 0, 0);
  v31 = sub_1157250((__int64 *)a2, v69, (__int64)v92);
LABEL_34:
  *(_QWORD *)(v5 + 520) = v31;
  v32 = sub_2434E10(v5, (__int64 *)a2);
  v33 = *(__int64 **)(a2 + 72);
  v95 = 257;
  v34 = v32;
  v35 = (__int64 **)sub_BCE3C0(v33, 0);
  v82 = sub_2436E50((__int64 *)a2, 0x30u, (unsigned __int64)v7, v35, (__int64)v94, 0, v92[0], 0);
  v36 = sub_AA4E30(*(_QWORD *)(a2 + 48));
  v37 = sub_AE5020(v36, *(_QWORD *)(v34 + 8));
  v95 = 257;
  v83 = v37;
  v38 = sub_BD2C40(80, unk_3F10A10);
  v40 = (__int64)v38;
  if ( v38 )
    sub_B4D3C0((__int64)v38, v34, v82, 0, v83, v39, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
    *(_QWORD *)(a2 + 88),
    v40,
    v94,
    *(_QWORD *)(a2 + 56),
    *(_QWORD *)(a2 + 64));
  if ( *(_QWORD *)a2 != *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8) )
  {
    v85 = v5;
    v41 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
    v42 = *(unsigned int **)a2;
    do
    {
      v43 = *((_QWORD *)v42 + 1);
      v44 = *v42;
      v42 += 4;
      sub_B99FD0(v40, v44, v43);
    }
    while ( (unsigned int *)v41 != v42 );
    v5 = v85;
  }
  v45 = sub_2A3B340(a2, v90, 8);
  v46 = sub_AA4E30(*(_QWORD *)(a2 + 48));
  v47 = sub_AE5020(v46, *(_QWORD *)(v45 + 8));
  v95 = 257;
  v84 = v47;
  v48 = sub_BD2C40(80, unk_3F10A10);
  v50 = (__int64)v48;
  if ( v48 )
    sub_B4D3C0((__int64)v48, v45, v81, 0, v84, v49, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
    *(_QWORD *)(a2 + 88),
    v50,
    v94,
    *(_QWORD *)(a2 + 56),
    *(_QWORD *)(a2 + 64));
  if ( *(_QWORD *)a2 != *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8) )
  {
    v86 = v5;
    v51 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
    v52 = *(unsigned int **)a2;
    do
    {
      v53 = *((_QWORD *)v52 + 1);
      v54 = *v52;
      v52 += 4;
      sub_B99FD0(v50, v54, v53);
    }
    while ( (unsigned int *)v51 != v52 );
    v5 = v86;
  }
  if ( !*(_QWORD *)(v5 + 512) )
  {
    if ( v7 )
    {
LABEL_60:
      v59 = *(_QWORD *)(v5 + 120);
      v95 = 259;
      v94[0] = "hwasan.shadow";
      v60 = sub_AD64C0(v59, 1, 0);
      v61 = *(_QWORD *)(v5 + 120);
      v62 = (_BYTE *)v60;
      v93 = 257;
      v63 = (_BYTE *)sub_AD64C0(v61, 0xFFFFFFFFLL, 0);
      v64 = (_BYTE *)sub_A82480((unsigned int **)a2, v7, v63, (__int64)v92);
      v65 = sub_929C50((unsigned int **)a2, v64, v62, (__int64)v94, 0, 0);
      v95 = 257;
      *(_QWORD *)(v5 + 512) = v65;
      *(_QWORD *)(v5 + 512) = sub_2436E50(
                                (__int64 *)a2,
                                0x30u,
                                v65,
                                *(__int64 ***)(v5 + 128),
                                (__int64)v94,
                                0,
                                v92[0],
                                0);
      return;
    }
    v7 = v90;
    if ( v81 )
    {
LABEL_58:
      if ( (unsigned int)(*(_DWORD *)(v5 + 56) - 3) > 2 )
        v7 = (_BYTE *)sub_2435400(
                        *(_BYTE *)(v5 + 160),
                        *(_DWORD *)(v5 + 176),
                        *(_QWORD *)(v5 + 184),
                        (__int64 *)a2,
                        (__int64)v7);
      goto LABEL_60;
    }
LABEL_54:
    if ( (unsigned int)(*(_DWORD *)(v5 + 56) - 3) <= 2 && *(_DWORD *)(v5 + 72) == 17 )
      v58 = sub_2A3A9B0(a2, 6);
    else
      v58 = *(_QWORD *)(v5 + 536);
    if ( !v7 )
    {
      v70 = *(_QWORD *)(a2 + 48);
      v71 = *(_QWORD *)(v5 + 120);
      v93 = 257;
      v72 = sub_AA4E30(v70);
      v73 = sub_AE5020(v72, v71);
      v95 = 257;
      v87 = v73;
      v74 = sub_BD2C40(80, unk_3F10A14);
      v7 = v74;
      if ( v74 )
      {
        sub_B4D190((__int64)v74, v71, v58, (__int64)v94, 0, v87, 0, 0);
        v75 = v80;
      }
      (*(void (__fastcall **)(_QWORD, _BYTE *, int *, _QWORD, _QWORD, __int64))(**(_QWORD **)(a2 + 88) + 16LL))(
        *(_QWORD *)(a2 + 88),
        v7,
        v92,
        *(_QWORD *)(a2 + 56),
        *(_QWORD *)(a2 + 64),
        v75);
      v76 = *(unsigned int **)a2;
      v77 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
      if ( *(_QWORD *)a2 != v77 )
      {
        do
        {
          v78 = *((_QWORD *)v76 + 1);
          v79 = *v76;
          v76 += 4;
          sub_B99FD0((__int64)v7, v79, v78);
        }
        while ( (unsigned int *)v77 != v76 );
      }
    }
    goto LABEL_58;
  }
}
