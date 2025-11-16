// Function: sub_922290
// Address: 0x922290
//
__int64 __fastcall sub_922290(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v5; // rcx
  unsigned __int64 v6; // rsi
  __int64 v7; // rbx
  __int64 *v8; // rdi
  __int64 v9; // rax
  unsigned __int64 v10; // rsi
  __int64 v11; // r14
  __int64 v12; // rcx
  __int64 v13; // rax
  unsigned __int64 v14; // rax
  int v15; // ebx
  int v16; // ecx
  __int64 v17; // rax
  char v18; // al
  __int16 v19; // cx
  __int64 v20; // rax
  int v21; // r9d
  __int64 v22; // r12
  __int64 v23; // rsi
  __int64 v24; // rdx
  unsigned int *v25; // r14
  unsigned int *v26; // rbx
  __int64 v27; // rdx
  __int64 v28; // r12
  int v29; // eax
  __int64 v30; // rdi
  _BOOL4 v31; // edx
  int v32; // ebx
  unsigned __int64 v34; // r12
  __int64 v35; // rax
  __int64 v36; // rdi
  _QWORD *v37; // rax
  __int64 v38; // rsi
  __int64 v39; // rcx
  __int64 v40; // rdx
  __int64 v41; // r13
  int v42; // eax
  __int64 v43; // rdi
  _BOOL4 v44; // edx
  int v45; // ebx
  __int64 *v46; // rdi
  __int64 v47; // rax
  unsigned __int64 v48; // rsi
  __int64 v49; // r12
  int v50; // ecx
  __int64 v51; // rdi
  unsigned __int64 v52; // rax
  __int64 v53; // rax
  unsigned __int8 v54; // al
  int v55; // r8d
  __int64 v56; // rax
  int v57; // r9d
  __int64 v58; // rbx
  __int64 v59; // r12
  unsigned int *v60; // r13
  unsigned int *v61; // r12
  __int64 v62; // rdx
  char *v63; // [rsp+18h] [rbp-A8h]
  __int64 v64; // [rsp+20h] [rbp-A0h]
  __int64 v65; // [rsp+28h] [rbp-98h]
  int v67; // [rsp+40h] [rbp-80h]
  __int16 v68; // [rsp+44h] [rbp-7Ch]
  int v69; // [rsp+44h] [rbp-7Ch]
  __int64 v70; // [rsp+48h] [rbp-78h]
  int v71; // [rsp+48h] [rbp-78h]
  int v72; // [rsp+48h] [rbp-78h]
  int v73; // [rsp+48h] [rbp-78h]
  int v74; // [rsp+5Ch] [rbp-64h] BYREF
  _QWORD v75[4]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v76; // [rsp+80h] [rbp-40h]

  v3 = a3;
  if ( (unsigned __int8)sub_920430(*(_QWORD *)(a2 + 32), a3, &v74) )
  {
    v6 = *(_QWORD *)(v3 + 120);
    v7 = v74;
    v76 = 259;
    v75[0] = "predef_tmp";
    v65 = sub_921CE0(a2, v6, (__int64)v75, v5);
    if ( (_DWORD)v7 == 4 )
    {
      v46 = *(__int64 **)(a2 + 32);
      v76 = 257;
      v47 = sub_90A810(v46, 9374, 0, 0);
      v48 = 0;
      if ( v47 )
        v48 = *(_QWORD *)(v47 + 24);
      v49 = sub_921880((unsigned int **)(a2 + 48), v48, v47, 0, 0, (__int64)v75, 0);
      v50 = unk_4D0463C;
      if ( unk_4D0463C )
        v50 = sub_90AA40(*(_QWORD *)(a2 + 32), v65);
      v51 = *(_QWORD *)(v3 + 120);
      if ( *(char *)(v51 + 142) >= 0 && *(_BYTE *)(v51 + 140) == 12 )
      {
        v73 = v50;
        LODWORD(v52) = sub_8D4AB0(v51);
        v50 = v73;
        v52 = (unsigned int)v52;
      }
      else
      {
        v52 = *(unsigned int *)(v51 + 136);
      }
      if ( v52 )
      {
        _BitScanReverse64(&v52, v52);
        v55 = (unsigned __int8)(63 - (v52 ^ 0x3F));
      }
      else
      {
        v71 = v50;
        v53 = sub_AA4E30(*(_QWORD *)(a2 + 96));
        v54 = sub_AE5020(v53, *(_QWORD *)(v49 + 8));
        v50 = v71;
        v55 = v54;
      }
      v69 = v55;
      v76 = 257;
      v72 = v50;
      v56 = sub_BD2C40(80, unk_3F10A10);
      v58 = v56;
      if ( v56 )
        sub_B4D3C0(v56, v49, v65, v72, v69, v57, 0, 0);
      v23 = v58;
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 136) + 16LL))(
        *(_QWORD *)(a2 + 136),
        v58,
        v75,
        *(_QWORD *)(a2 + 104),
        *(_QWORD *)(a2 + 112));
      v59 = 4LL * *(unsigned int *)(a2 + 56);
      v60 = *(unsigned int **)(a2 + 48);
      v61 = &v60[v59];
      while ( v61 != v60 )
      {
        v62 = *((_QWORD *)v60 + 1);
        v23 = *v60;
        v60 += 4;
        sub_B99FD0(v58, v23, v62);
      }
    }
    else
    {
      v70 = 0;
      v64 = v3;
      v63 = (char *)&unk_3F109E0 + 12 * v7;
      do
      {
        v8 = *(__int64 **)(a2 + 32);
        v76 = 257;
        v9 = sub_90A810(v8, *(unsigned int *)&v63[4 * v70], 0, 0);
        v10 = 0;
        if ( v9 )
          v10 = *(_QWORD *)(v9 + 24);
        v11 = sub_921880((unsigned int **)(a2 + 48), v10, v9, 0, 0, (__int64)v75, 0);
        v76 = 257;
        v13 = sub_91A390(*(_QWORD *)(a2 + 32) + 8LL, *(_QWORD *)(v64 + 120), 0, v12);
        v14 = sub_9213A0((unsigned int **)(a2 + 48), v13, v65, 0, v70, (__int64)v75, 0);
        v15 = v14;
        v16 = unk_4D0463C;
        if ( unk_4D0463C )
          v16 = sub_90AA40(*(_QWORD *)(a2 + 32), v14);
        v67 = v16;
        v17 = sub_AA4E30(*(_QWORD *)(a2 + 96));
        v18 = sub_AE5020(v17, *(_QWORD *)(v11 + 8));
        HIBYTE(v19) = HIBYTE(v68);
        LOBYTE(v19) = v18;
        v68 = v19;
        v76 = 257;
        v20 = sub_BD2C40(80, unk_3F10A10);
        v22 = v20;
        if ( v20 )
          sub_B4D3C0(v20, v11, v15, v67, (unsigned __int8)v68, v21, 0, 0);
        v23 = v22;
        (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 136) + 16LL))(
          *(_QWORD *)(a2 + 136),
          v22,
          v75,
          *(_QWORD *)(a2 + 104),
          *(_QWORD *)(a2 + 112));
        v25 = *(unsigned int **)(a2 + 48);
        v26 = &v25[4 * *(unsigned int *)(a2 + 56)];
        while ( v26 != v25 )
        {
          v27 = *((_QWORD *)v25 + 1);
          v23 = *v25;
          v25 += 4;
          sub_B99FD0(v22, v23, v27);
        }
        ++v70;
      }
      while ( v70 != 3 );
      v3 = v64;
    }
    v28 = *(_QWORD *)(v3 + 120);
    v29 = sub_91CB50(v3, v23, v24);
    v30 = *(_QWORD *)(v3 + 120);
    v31 = 0;
    v32 = v29;
    if ( (*(_BYTE *)(v30 + 140) & 0xFB) == 8 )
      v31 = (sub_8D4C10(v30, dword_4F077C4 != 2) & 2) != 0;
    *(_DWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = v65;
    *(_QWORD *)(a1 + 16) = v28;
    *(_DWORD *)(a1 + 48) = v31;
    *(_DWORD *)(a1 + 24) = v32;
  }
  else
  {
    v34 = sub_916430(*(__int64 **)(a2 + 32), v3, 0);
    v35 = *(_QWORD *)(a2 + 32);
    v36 = v35 + 512;
    v37 = *(_QWORD **)(v35 + 520);
    if ( !v37 )
      goto LABEL_28;
    v38 = v36;
    do
    {
      while ( 1 )
      {
        v39 = v37[2];
        v40 = v37[3];
        if ( v37[4] >= v34 )
          break;
        v37 = (_QWORD *)v37[3];
        if ( !v40 )
          goto LABEL_23;
      }
      v38 = (__int64)v37;
      v37 = (_QWORD *)v37[2];
    }
    while ( v39 );
LABEL_23:
    if ( v36 == v38 || *(_QWORD *)(v38 + 32) > v34 )
    {
LABEL_28:
      v38 = v34;
      v34 = sub_92CAE0(a2, v34, v3 + 64);
    }
    v41 = *(_QWORD *)(v3 + 120);
    v42 = sub_91CB50(v3, v38, v40);
    v43 = *(_QWORD *)(v3 + 120);
    v44 = 0;
    v45 = v42;
    if ( (*(_BYTE *)(v43 + 140) & 0xFB) == 8 )
      v44 = (sub_8D4C10(v43, dword_4F077C4 != 2) & 2) != 0;
    *(_DWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = v34;
    *(_QWORD *)(a1 + 16) = v41;
    *(_DWORD *)(a1 + 48) = v44;
    *(_DWORD *)(a1 + 24) = v45;
  }
  return a1;
}
