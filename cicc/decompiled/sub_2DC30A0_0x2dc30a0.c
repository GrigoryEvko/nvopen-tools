// Function: sub_2DC30A0
// Address: 0x2dc30a0
//
void __fastcall sub_2DC30A0(__int64 a1)
{
  bool v2; // zf
  __int64 v3; // rdi
  __int16 v4; // dx
  __int64 v5; // r8
  char v6; // al
  char v7; // dl
  __int16 v8; // cx
  __int64 v9; // rax
  _QWORD *v10; // rdi
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // r13
  __int64 v18; // rcx
  __int64 v19; // r14
  int v20; // eax
  int v21; // eax
  unsigned int v22; // edx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // r13
  _QWORD *v27; // rax
  __int64 v28; // r14
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // rdi
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // rbx
  __int64 v36; // r13
  __int64 v37; // rdx
  unsigned int v38; // esi
  __int64 v39; // rdi
  __int64 v40; // rax
  __int16 v41; // dx
  __int64 v42; // r8
  char v43; // al
  char v44; // dl
  __int16 v45; // cx
  _QWORD *v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // r13
  __int64 v50; // r15
  __int64 v51; // r14
  int v52; // eax
  int v53; // eax
  unsigned int v54; // edx
  __int64 v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rdx
  __int64 v58; // r13
  _QWORD *v59; // rax
  __int64 v60; // r14
  __int64 v61; // rdi
  __int64 v62; // rcx
  __int64 v63; // rbx
  __int64 v64; // r13
  __int64 v65; // rdx
  unsigned int v66; // esi
  __int64 v67; // [rsp+8h] [rbp-68h]
  unsigned __int64 v68[4]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v69; // [rsp+30h] [rbp-40h]

  v2 = *(_BYTE *)(a1 + 104) == 0;
  v3 = *(_QWORD *)(a1 + 8);
  if ( v2 )
  {
    v5 = sub_AA5190(v3);
    if ( v5 )
    {
      v6 = v4;
      v7 = HIBYTE(v4);
    }
    else
    {
      v7 = 0;
      v6 = 0;
    }
    LOBYTE(v8) = v6;
    HIBYTE(v8) = v7;
    sub_A88F30(a1 + 128, *(_QWORD *)(a1 + 8), v5, v8);
    v69 = 257;
    v9 = sub_92B530((unsigned int **)(a1 + 128), 0x24u, *(_QWORD *)(a1 + 16), *(_BYTE **)(a1 + 24), (__int64)v68);
    v10 = *(_QWORD **)(a1 + 200);
    v69 = 257;
    v11 = v9;
    v12 = sub_BCB2D0(v10);
    v13 = sub_ACD640(v12, 1, 0);
    v14 = sub_BCB2D0(*(_QWORD **)(a1 + 200));
    v15 = sub_AD62B0(v14);
    v16 = sub_B36550((unsigned int **)(a1 + 128), v11, v15, v13, (__int64)v68, 0);
    v17 = *(_QWORD *)(a1 + 96);
    v18 = *(_QWORD *)(a1 + 8);
    v19 = v16;
    v20 = *(_DWORD *)(v17 + 4) & 0x7FFFFFF;
    if ( v20 == *(_DWORD *)(v17 + 72) )
    {
      v67 = *(_QWORD *)(a1 + 8);
      sub_B48D90(*(_QWORD *)(a1 + 96));
      v18 = v67;
      v20 = *(_DWORD *)(v17 + 4) & 0x7FFFFFF;
    }
    v21 = (v20 + 1) & 0x7FFFFFF;
    v22 = v21 | *(_DWORD *)(v17 + 4) & 0xF8000000;
    v23 = *(_QWORD *)(v17 - 8) + 32LL * (unsigned int)(v21 - 1);
    *(_DWORD *)(v17 + 4) = v22;
    if ( *(_QWORD *)v23 )
    {
      v24 = *(_QWORD *)(v23 + 8);
      **(_QWORD **)(v23 + 16) = v24;
      if ( v24 )
        *(_QWORD *)(v24 + 16) = *(_QWORD *)(v23 + 16);
    }
    *(_QWORD *)v23 = v19;
    if ( v19 )
    {
      v25 = *(_QWORD *)(v19 + 16);
      *(_QWORD *)(v23 + 8) = v25;
      if ( v25 )
        *(_QWORD *)(v25 + 16) = v23 + 8;
      *(_QWORD *)(v23 + 16) = v19 + 16;
      *(_QWORD *)(v19 + 16) = v23;
    }
    *(_QWORD *)(*(_QWORD *)(v17 - 8)
              + 32LL * *(unsigned int *)(v17 + 72)
              + 8LL * ((*(_DWORD *)(v17 + 4) & 0x7FFFFFFu) - 1)) = v18;
    v26 = *(_QWORD *)(a1 + 88);
    v27 = sub_BD2C40(72, 1u);
    v28 = (__int64)v27;
    if ( v27 )
      sub_B4C8F0((__int64)v27, v26, 1u, 0, 0);
    v29 = *(_QWORD *)(a1 + 184);
    v30 = *(_QWORD *)(a1 + 192);
    v31 = *(_QWORD *)(a1 + 216);
    v69 = 257;
    (*(void (__fastcall **)(__int64, __int64, unsigned __int64 *, __int64, __int64))(*(_QWORD *)v31 + 16LL))(
      v31,
      v28,
      v68,
      v29,
      v30);
    v35 = *(_QWORD *)(a1 + 128);
    v36 = v35 + 16LL * *(unsigned int *)(a1 + 136);
    while ( v36 != v35 )
    {
      v37 = *(_QWORD *)(v35 + 8);
      v38 = *(_DWORD *)v35;
      v35 += 16;
      sub_B99FD0(v28, v38, v37);
    }
  }
  else
  {
    v42 = sub_AA5190(v3);
    if ( v42 )
    {
      v43 = v41;
      v44 = HIBYTE(v41);
    }
    else
    {
      v44 = 0;
      v43 = 0;
    }
    LOBYTE(v45) = v43;
    HIBYTE(v45) = v44;
    sub_A88F30(a1 + 128, *(_QWORD *)(a1 + 8), v42, v45);
    v46 = (_QWORD *)sub_BD5C60(*(_QWORD *)a1);
    v47 = sub_BCB2D0(v46);
    v48 = sub_ACD640(v47, 1, 0);
    v49 = *(_QWORD *)(a1 + 96);
    v50 = *(_QWORD *)(a1 + 8);
    v51 = v48;
    v52 = *(_DWORD *)(v49 + 4) & 0x7FFFFFF;
    if ( v52 == *(_DWORD *)(v49 + 72) )
    {
      sub_B48D90(*(_QWORD *)(a1 + 96));
      v52 = *(_DWORD *)(v49 + 4) & 0x7FFFFFF;
    }
    v53 = (v52 + 1) & 0x7FFFFFF;
    v54 = v53 | *(_DWORD *)(v49 + 4) & 0xF8000000;
    v55 = *(_QWORD *)(v49 - 8) + 32LL * (unsigned int)(v53 - 1);
    *(_DWORD *)(v49 + 4) = v54;
    if ( *(_QWORD *)v55 )
    {
      v56 = *(_QWORD *)(v55 + 8);
      **(_QWORD **)(v55 + 16) = v56;
      if ( v56 )
        *(_QWORD *)(v56 + 16) = *(_QWORD *)(v55 + 16);
    }
    *(_QWORD *)v55 = v51;
    if ( v51 )
    {
      v57 = *(_QWORD *)(v51 + 16);
      *(_QWORD *)(v55 + 8) = v57;
      if ( v57 )
        *(_QWORD *)(v57 + 16) = v55 + 8;
      *(_QWORD *)(v55 + 16) = v51 + 16;
      *(_QWORD *)(v51 + 16) = v55;
    }
    *(_QWORD *)(*(_QWORD *)(v49 - 8)
              + 32LL * *(unsigned int *)(v49 + 72)
              + 8LL * ((*(_DWORD *)(v49 + 4) & 0x7FFFFFFu) - 1)) = v50;
    v58 = *(_QWORD *)(a1 + 88);
    v59 = sub_BD2C40(72, 1u);
    v60 = (__int64)v59;
    if ( v59 )
      sub_B4C8F0((__int64)v59, v58, 1u, 0, 0);
    v61 = *(_QWORD *)(a1 + 216);
    v62 = *(_QWORD *)(a1 + 184);
    v69 = 257;
    (*(void (__fastcall **)(__int64, __int64, unsigned __int64 *, __int64, _QWORD))(*(_QWORD *)v61 + 16LL))(
      v61,
      v60,
      v68,
      v62,
      *(_QWORD *)(a1 + 192));
    v63 = *(_QWORD *)(a1 + 128);
    v64 = v63 + 16LL * *(unsigned int *)(a1 + 136);
    while ( v64 != v63 )
    {
      v65 = *(_QWORD *)(v63 + 8);
      v66 = *(_DWORD *)v63;
      v63 += 16;
      sub_B99FD0(v60, v66, v65);
    }
  }
  v39 = *(_QWORD *)(a1 + 120);
  if ( v39 )
  {
    v40 = *(_QWORD *)(a1 + 88);
    v68[0] = *(_QWORD *)(a1 + 8);
    v68[1] = v40 & 0xFFFFFFFFFFFFFFFBLL;
    sub_FFB3D0(v39, v68, 1, v32, v33, v34);
  }
}
