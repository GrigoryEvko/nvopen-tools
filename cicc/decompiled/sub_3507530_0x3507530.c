// Function: sub_3507530
// Address: 0x3507530
//
void __fastcall sub_3507530(__int64 *a1, __int64 a2, char a3)
{
  _QWORD *v4; // r14
  __int64 v5; // r15
  __int64 *v6; // rbx
  __int64 v7; // rax
  int v8; // esi
  unsigned __int64 v9; // r14
  char v10; // al
  __int64 v11; // rsi
  int v12; // eax
  __int64 *v13; // rax
  __int64 v14; // r11
  __int64 v15; // rcx
  _QWORD *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  unsigned __int64 v20; // r9
  __int64 *v21; // r13
  __int64 v22; // rdx
  _QWORD *v23; // r15
  _QWORD *v24; // rbx
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // rdi
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // rdx
  __int64 v31; // rdx
  _QWORD *v32; // rax
  __int64 v33; // r8
  __int64 v34; // [rsp+0h] [rbp-350h]
  __int64 v36; // [rsp+10h] [rbp-340h]
  __int64 v37; // [rsp+10h] [rbp-340h]
  __int64 v38; // [rsp+10h] [rbp-340h]
  __int64 v39; // [rsp+18h] [rbp-338h]
  __int64 v40; // [rsp+18h] [rbp-338h]
  _QWORD *v41; // [rsp+18h] [rbp-338h]
  __int64 v43; // [rsp+28h] [rbp-328h]
  __int64 v44; // [rsp+28h] [rbp-328h]
  _QWORD *v45; // [rsp+30h] [rbp-320h]
  __int64 v46; // [rsp+30h] [rbp-320h]
  int v47; // [rsp+3Ch] [rbp-314h]
  __int64 v48; // [rsp+40h] [rbp-310h]
  __int64 v49; // [rsp+40h] [rbp-310h]
  __int64 v50; // [rsp+48h] [rbp-308h]
  __int64 v51; // [rsp+48h] [rbp-308h]
  _QWORD v52[2]; // [rsp+50h] [rbp-300h] BYREF
  __int64 (__fastcall *v53)(unsigned __int64 *, const __m128i **, int); // [rsp+60h] [rbp-2F0h]
  unsigned __int64 *(__fastcall *v54)(unsigned __int64 **, __int64); // [rsp+68h] [rbp-2E8h]
  __int64 v55; // [rsp+70h] [rbp-2E0h]
  _BYTE *v56; // [rsp+78h] [rbp-2D8h]
  __int64 v57; // [rsp+80h] [rbp-2D0h]
  _BYTE v58[48]; // [rsp+88h] [rbp-2C8h] BYREF
  int v59; // [rsp+B8h] [rbp-298h]
  __int64 v60; // [rsp+C0h] [rbp-290h]
  _QWORD *v61; // [rsp+C8h] [rbp-288h]
  __int64 v62; // [rsp+D0h] [rbp-280h]
  unsigned int v63; // [rsp+D8h] [rbp-278h]
  _QWORD *v64; // [rsp+E0h] [rbp-270h]
  __int64 v65; // [rsp+E8h] [rbp-268h]
  _QWORD v66[3]; // [rsp+F0h] [rbp-260h] BYREF
  _BYTE *v67; // [rsp+108h] [rbp-248h]
  __int64 v68; // [rsp+110h] [rbp-240h]
  _BYTE v69[568]; // [rsp+118h] [rbp-238h] BYREF

  v4 = (_QWORD *)a1[1];
  v5 = a1[2];
  v6 = (__int64 *)a1[4];
  v43 = (__int64)v4;
  v7 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*v4 + 16LL) + 200LL))(*(_QWORD *)(*v4 + 16LL));
  v8 = *(_DWORD *)(a2 + 112);
  v45 = (_QWORD *)v7;
  v47 = v8;
  if ( v8 < 0 )
    v9 = *(_QWORD *)(v4[7] + 16LL * (v8 & 0x7FFFFFFF) + 8);
  else
    v9 = *(_QWORD *)(v4[38] + 8LL * (unsigned int)v8);
  if ( !v9 )
    goto LABEL_20;
  while ( (*(_BYTE *)(v9 + 4) & 8) != 0 )
  {
    v9 = *(_QWORD *)(v9 + 32);
    if ( !v9 )
      goto LABEL_20;
  }
LABEL_5:
  if ( (*(_BYTE *)(v9 + 3) & 0x10) == 0 )
  {
    v10 = *(_BYTE *)(v9 + 4);
    if ( (v10 & 1) != 0 || (v10 & 2) != 0 )
      goto LABEL_19;
  }
  v11 = *(_QWORD *)(a2 + 104);
  v12 = (*(_DWORD *)v9 >> 8) & 0xFFF;
  if ( v11 )
  {
    if ( !v12 )
    {
      v14 = sub_2EBF1E0(v43, v47);
      v15 = v30;
      if ( *(_QWORD *)(a2 + 104) )
        goto LABEL_11;
LABEL_48:
      if ( *(_DWORD *)(a2 + 8) )
      {
        v37 = v15;
        v40 = v14;
        v51 = sub_2EBF1E0(v43, v47);
        v49 = v31;
        v32 = (_QWORD *)sub_A777F0(0x80u, v6);
        v14 = v40;
        v15 = v37;
        if ( v32 )
        {
          v32[12] = 0;
          *v32 = v32 + 2;
          v32[1] = 0x200000000LL;
          v32[8] = v32 + 10;
          v32[9] = 0x200000000LL;
          v34 = v37;
          v38 = v40;
          v41 = v32;
          sub_2F68500((__int64)v32, (__int64 *)a2, v6, v15, v33);
          v32 = v41;
          v15 = v34;
          v14 = v38;
          v41[14] = v51;
          v41[13] = 0;
          v41[15] = v49;
        }
        v32[13] = *(_QWORD *)(a2 + 104);
        *(_QWORD *)(a2 + 104) = v32;
      }
      goto LABEL_11;
    }
  }
  else if ( !v12 || !a3 )
  {
    if ( (*(_BYTE *)(v9 + 3) & 0x10) == 0 )
      goto LABEL_19;
    goto LABEL_46;
  }
  v13 = (__int64 *)(v45[34] + 16LL * ((*(_DWORD *)v9 >> 8) & 0xFFF));
  v14 = *v13;
  v15 = v13[1];
  if ( !v11 )
    goto LABEL_48;
LABEL_11:
  v48 = v15;
  v50 = v14;
  v53 = 0;
  v16 = (_QWORD *)sub_22077B0(0x18u);
  if ( v16 )
  {
    *v16 = v9;
    v16[1] = v5;
    v16[2] = v6;
  }
  v52[0] = v16;
  v54 = sub_3506EB0;
  v53 = sub_3506D20;
  sub_2E0C490(a2, v6, v50, v48, (unsigned __int64)v52, v5, v45, 0);
  if ( v53 )
    v53(v52, (const __m128i **)v52, 3);
  if ( (*(_BYTE *)(v9 + 3) & 0x10) != 0 && !*(_QWORD *)(a2 + 104) )
LABEL_46:
    sub_3506D90(v5, v6, a2, v9);
LABEL_19:
  while ( 1 )
  {
    v9 = *(_QWORD *)(v9 + 32);
    if ( !v9 )
      break;
    if ( (*(_BYTE *)(v9 + 4) & 8) == 0 )
      goto LABEL_5;
  }
LABEL_20:
  sub_2E0AF60(a2);
  v21 = *(__int64 **)(a2 + 104);
  v44 = *a1;
  v46 = a1[3];
  if ( v21 )
  {
    v39 = v5;
    v36 = (__int64)v6;
    do
    {
      v52[0] = 0;
      v52[1] = 0;
      v56 = v58;
      v57 = 0x600000000LL;
      v53 = 0;
      v64 = v66;
      v68 = 0x1000000000LL;
      v54 = 0;
      v55 = 0;
      v59 = 0;
      v60 = 0;
      v61 = 0;
      v62 = 0;
      v63 = 0;
      v65 = 0;
      v66[0] = 0;
      v66[1] = 0;
      v67 = v69;
      sub_2E1DCC0((__int64)v52, v44, v39, v46, v36, v20);
      sub_3507070(v52, v21, v47, v21[14], v21[15], a2);
      if ( v67 != v69 )
        _libc_free((unsigned __int64)v67);
      if ( v64 != v66 )
        _libc_free((unsigned __int64)v64);
      v22 = v63;
      if ( v63 )
      {
        v23 = v61;
        v24 = &v61[19 * v63];
        do
        {
          if ( *v23 != -4096 && *v23 != -8192 )
          {
            v25 = v23[10];
            if ( (_QWORD *)v25 != v23 + 12 )
              _libc_free(v25);
            v26 = v23[1];
            if ( (_QWORD *)v26 != v23 + 3 )
              _libc_free(v26);
          }
          v23 += 19;
        }
        while ( v24 != v23 );
        v22 = v63;
      }
      sub_C7D6A0((__int64)v61, 152 * v22, 8);
      if ( v56 != v58 )
        _libc_free((unsigned __int64)v56);
      v21 = (__int64 *)v21[13];
    }
    while ( v21 );
    *(_DWORD *)(a2 + 72) = 0;
    *(_DWORD *)(a2 + 8) = 0;
    sub_3507470(a1, a2, v27, v28, v29, v20);
  }
  else
  {
    sub_2E1D8A0((__int64)a1, *a1, v17, v18, v19, v20);
    sub_3507070(a1, (__int64 *)a2, v47, -1, -1, 0);
  }
}
