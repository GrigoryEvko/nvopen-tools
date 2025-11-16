// Function: sub_34CE4D0
// Address: 0x34ce4d0
//
__int64 __fastcall sub_34CE4D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v10; // r11
  __int64 v11; // r9
  __int64 v12; // r15
  char v13; // al
  __int64 (__fastcall *v15)(__int64, __int64); // rax
  __int64 v17; // rax
  __int64 v18; // r11
  void *v19; // r9
  __int64 v20; // rdi
  __int64 (__fastcall *v21)(__int64, __int64, __int64, __int64); // rax
  __int64 v22; // rax
  __int64 v23; // r14
  __int64 v24; // rdx
  __int64 v25; // rdx
  _QWORD *v26; // rdi
  __int64 v27; // rsi
  __int64 (__fastcall *v28)(__int64, __int64, __int64, __int64, __int64); // rax
  __int64 v29; // rax
  __int64 (__fastcall *v30)(_QWORD *, __int64, __int64, __int64, __int64, __int64); // rax
  __int64 v31; // rax
  __int64 v32; // r15
  int v33; // esi
  __int64 v34; // rax
  __int64 v35; // r13
  __int64 v36; // rax
  __int64 v37; // r13
  char v38; // al
  __int64 v39; // rdi
  __int64 v40; // r13
  void (__fastcall *v41)(__int64); // rax
  __int64 (__fastcall *v42)(__int64, __int64); // rax
  __int64 v43; // rax
  unsigned int v44; // eax
  const char *v45; // rbx
  __int64 v46; // rdx
  __int64 v47; // r13
  unsigned int v48; // r14d
  __int64 v49; // rax
  __int64 v50; // rbx
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // [rsp-8h] [rbp-E8h]
  _DWORD *v54; // [rsp+8h] [rbp-D8h]
  __int64 v55; // [rsp+10h] [rbp-D0h]
  __int64 v56; // [rsp+10h] [rbp-D0h]
  __int64 v57; // [rsp+18h] [rbp-C8h]
  __int64 v58; // [rsp+20h] [rbp-C0h]
  void *v59; // [rsp+20h] [rbp-C0h]
  __int64 v60; // [rsp+20h] [rbp-C0h]
  void *v61; // [rsp+20h] [rbp-C0h]
  __int64 v62; // [rsp+28h] [rbp-B8h]
  __int64 v63; // [rsp+28h] [rbp-B8h]
  _DWORD *v64; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v65; // [rsp+38h] [rbp-A8h] BYREF
  void *v66[4]; // [rsp+40h] [rbp-A0h] BYREF
  __int16 v67; // [rsp+60h] [rbp-80h]
  unsigned __int64 v68[2]; // [rsp+70h] [rbp-70h] BYREF
  __int64 v69; // [rsp+80h] [rbp-60h] BYREF
  char v70; // [rsp+90h] [rbp-50h]
  char v71; // [rsp+91h] [rbp-4Fh]

  v10 = *(_QWORD *)(a2 + 664);
  v11 = *(_QWORD *)(a2 + 672);
  v62 = *(_QWORD *)(a2 + 680);
  if ( (_DWORD)a5 != 1 )
  {
    if ( (_DWORD)a5 == 2 )
    {
      v40 = *(_QWORD *)(a2 + 8);
      v12 = sub_3717C80(a6, a2, a3, a4, a5, v11);
      v41 = *(void (__fastcall **)(__int64))(v40 + 184);
      if ( v41 )
        v41(v12);
      goto LABEL_4;
    }
    v12 = 0;
    if ( (_DWORD)a5 )
    {
LABEL_4:
      v13 = *(_BYTE *)(a1 + 8);
      *(_QWORD *)a1 = v12;
      *(_BYTE *)(a1 + 8) = v13 & 0xFC | 2;
      return a1;
    }
    v25 = *(_QWORD *)(a2 + 656);
    v26 = *(_QWORD **)(a2 + 8);
    v27 = *(unsigned int *)(v25 + 176);
    if ( *(_BYTE *)(a2 + 988) )
      v27 = *(unsigned int *)(a2 + 984);
    v28 = (__int64 (__fastcall *)(__int64, __int64, __int64, __int64, __int64))v26[17];
    if ( v28 )
    {
      v55 = v10;
      v60 = v11;
      v29 = v28(a2 + 512, v27, v25, v11, v10);
      v26 = *(_QWORD **)(a2 + 8);
      v11 = v60;
      v57 = v29;
      v10 = v55;
    }
    else
    {
      v57 = 0;
    }
    v61 = 0;
    if ( (*(_BYTE *)(a2 + 977) & 2) != 0 )
    {
      v42 = (__int64 (__fastcall *)(__int64, __int64))v26[18];
      if ( v42 )
      {
        v56 = v10;
        v43 = v42(v11, a6);
        v26 = *(_QWORD **)(a2 + 8);
        v10 = v56;
        v61 = (void *)v43;
      }
    }
    v30 = (__int64 (__fastcall *)(_QWORD *, __int64, __int64, __int64, __int64, __int64))v26[13];
    if ( v30 )
      v63 = v30(v26, v62, v10, a2 + 976, a5, v11);
    else
      v63 = 0;
    v31 = sub_22077B0(0x70u);
    v32 = v31;
    if ( !v31 )
    {
LABEL_35:
      v39 = *(_QWORD *)(a2 + 8);
      v65 = v32;
      v68[0] = v63;
      v66[0] = v61;
      v12 = sub_C0D420(v39, a6, &v65, v57, (__int64)v66, (__int64)v68);
      if ( v65 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v65 + 8LL))(v65);
      if ( v66[0] )
        (*(void (__fastcall **)(void *))(*(_QWORD *)v66[0] + 8LL))(v66[0]);
      if ( v68[0] )
        (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v68[0] + 8LL))(v68[0]);
      goto LABEL_4;
    }
    *(_DWORD *)(v31 + 8) = 0;
    v33 = *(_DWORD *)(a3 + 44);
    *(_BYTE *)(v31 + 40) = 0;
    *(_DWORD *)(v31 + 44) = 1;
    *(_QWORD *)(v31 + 32) = 0;
    *(_QWORD *)(v31 + 24) = 0;
    *(_QWORD *)(v31 + 16) = 0;
    *(_QWORD *)(v31 + 56) = 0;
    *(_QWORD *)(v31 + 80) = 0;
    *(_QWORD *)v31 = &unk_49DC840;
    *(_QWORD *)(v31 + 72) = v31 + 96;
    v34 = *(_QWORD *)(a3 + 16);
    *(_QWORD *)(v32 + 88) = 4;
    *(_BYTE *)(v32 + 104) = 0;
    *(_QWORD *)(v32 + 48) = a3;
    if ( v33 && !v34 )
    {
      v35 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a3 + 88LL))(a3);
      v51 = *(_QWORD *)(v32 + 16);
      if ( v35 )
      {
        if ( *(_QWORD *)(v32 + 32) != v51 )
          sub_CB5AE0((__int64 *)v32);
        goto LABEL_31;
      }
      if ( *(_QWORD *)(v32 + 32) != v51 )
        sub_CB5AE0((__int64 *)v32);
    }
    else
    {
      v35 = *(_QWORD *)(a3 + 24) - v34;
      if ( v35 )
      {
LABEL_31:
        v36 = sub_2207820(v35);
        sub_CB5980(v32, v36, v35, 1);
LABEL_32:
        v37 = *(_QWORD *)(v32 + 48);
        if ( *(_QWORD *)(v37 + 32) != *(_QWORD *)(v37 + 16) )
          sub_CB5AE0(*(__int64 **)(v32 + 48));
        sub_CB5980(v37, 0, 0, 0);
        v38 = *(_BYTE *)(*(_QWORD *)(v32 + 48) + 40LL);
        *(_QWORD *)(v32 + 64) = 0;
        *(_BYTE *)(v32 + 40) = v38;
        goto LABEL_35;
      }
    }
    sub_CB5980(v32, 0, 0, 0);
    goto LABEL_32;
  }
  v15 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)(a2 + 8) + 144LL);
  if ( v15 && (v58 = *(_QWORD *)(a2 + 664), v17 = v15(v11, a6), v18 = v58, (v19 = (void *)v17) != 0) )
  {
    v20 = *(_QWORD *)(a2 + 8);
    v21 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(v20 + 104);
    if ( v21 )
    {
      v59 = v19;
      v22 = v21(v20, v62, v18, a2 + 976);
      if ( v22 )
      {
        v54 = (_DWORD *)v22;
        v67 = 260;
        v66[0] = (void *)(a2 + 512);
        sub_CC9F70((__int64)v68, v66);
        v23 = *(_QWORD *)(a2 + 8);
        v66[0] = v59;
        if ( a4 )
          sub_106E0D0(&v65, v54, a3, a4);
        else
          sub_106DB90(&v65, v54, a3);
        v64 = v54;
        v12 = sub_C0D2B0(v23, (__int64)v68, a6, (__int64)&v64, (__int64)&v65, (__int64)v66, v62);
        v24 = v53;
        if ( v64 )
          (*(void (__fastcall **)(_DWORD *, unsigned __int64 *))(*(_QWORD *)v64 + 8LL))(v64, v68);
        if ( v65 )
          (*(void (__fastcall **)(__int64, unsigned __int64 *, __int64))(*(_QWORD *)v65 + 8LL))(v65, v68, v24);
        if ( v66[0] )
          (*(void (__fastcall **)(void *, unsigned __int64 *, __int64))(*(_QWORD *)v66[0] + 8LL))(v66[0], v68, v24);
        if ( (__int64 *)v68[0] != &v69 )
          j_j___libc_free_0(v68[0]);
        goto LABEL_4;
      }
    }
    v44 = sub_C63BB0();
    v45 = "createMCAsmBackend failed";
    v71 = 1;
    v47 = v52;
  }
  else
  {
    v44 = sub_C63BB0();
    v71 = 1;
    v45 = "createMCCodeEmitter failed";
    v47 = v46;
  }
  v68[0] = (unsigned __int64)v45;
  v48 = v44;
  v70 = 3;
  v49 = sub_22077B0(0x40u);
  v50 = v49;
  if ( v49 )
    sub_C63EB0(v49, (__int64)v68, v48, v47);
  *(_BYTE *)(a1 + 8) |= 3u;
  *(_QWORD *)a1 = v50 & 0xFFFFFFFFFFFFFFFELL;
  return a1;
}
