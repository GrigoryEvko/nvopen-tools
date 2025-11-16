// Function: sub_2FDD7A0
// Address: 0x2fdd7a0
//
__int64 __fastcall sub_2FDD7A0(__int64 a1, __int64 a2, char a3, unsigned int a4, unsigned int a5)
{
  __int64 v5; // r12
  int v6; // esi
  __int64 v7; // rax
  __int64 v9; // rdx
  int v10; // r8d
  __int64 v11; // r10
  __int64 v12; // r13
  __int64 v13; // rbx
  __int64 v14; // rdi
  __int64 v15; // r11
  int v16; // r9d
  int v17; // r14d
  unsigned __int8 v18; // al
  char v19; // r11
  char v20; // r15
  char v21; // al
  __int16 v22; // dx
  _QWORD *v24; // rax
  _QWORD *v25; // rax
  unsigned __int8 v26; // al
  unsigned __int8 v27; // al
  unsigned __int16 *v28; // rax
  unsigned __int16 v29; // ax
  unsigned __int16 *v30; // rax
  unsigned __int16 v31; // ax
  _QWORD *v32; // rax
  _QWORD *v33; // rax
  unsigned int v34; // [rsp+8h] [rbp-78h]
  int v35; // [rsp+Ch] [rbp-74h]
  unsigned int v36; // [rsp+Ch] [rbp-74h]
  __int64 v37; // [rsp+10h] [rbp-70h]
  int v38; // [rsp+10h] [rbp-70h]
  __int64 v39; // [rsp+18h] [rbp-68h]
  __int64 v40; // [rsp+18h] [rbp-68h]
  __int64 v41; // [rsp+20h] [rbp-60h]
  int v42; // [rsp+28h] [rbp-58h]
  int v43; // [rsp+2Ch] [rbp-54h]
  int v44; // [rsp+2Ch] [rbp-54h]
  unsigned __int8 v45; // [rsp+30h] [rbp-50h]
  int v46; // [rsp+30h] [rbp-50h]
  int v48; // [rsp+34h] [rbp-4Ch]
  __int16 v49; // [rsp+34h] [rbp-4Ch]
  unsigned int v50; // [rsp+38h] [rbp-48h]
  unsigned __int8 v51; // [rsp+3Dh] [rbp-43h]
  char v52; // [rsp+3Eh] [rbp-42h]
  bool v53; // [rsp+3Fh] [rbp-41h]
  __int16 v54; // [rsp+40h] [rbp-40h]
  __int16 v55; // [rsp+44h] [rbp-3Ch]
  bool v56; // [rsp+48h] [rbp-38h]
  char v57; // [rsp+49h] [rbp-37h]
  char v58; // [rsp+4Ah] [rbp-36h]
  char v59; // [rsp+4Bh] [rbp-35h]
  __int16 v61; // [rsp+4Ch] [rbp-34h]
  int v62; // [rsp+4Ch] [rbp-34h]
  int v63; // [rsp+4Ch] [rbp-34h]

  v5 = a2;
  v6 = *(unsigned __int8 *)(*(_QWORD *)(a2 + 16) + 4LL);
  v7 = *(_QWORD *)(v5 + 32);
  if ( v6 )
  {
    if ( *(_BYTE *)v7 )
      return 0;
    v9 = a4;
    v10 = *(_DWORD *)(v7 + 8);
    v11 = a5;
    v12 = 40LL * a4;
    v13 = 40LL * a5;
    v14 = v7 + v12;
    v15 = v7 + v13;
    v16 = *(_DWORD *)(v7 + v12 + 8);
    v17 = *(_DWORD *)(v7 + v13 + 8);
    v43 = (*(_DWORD *)v7 >> 8) & 0xFFF;
  }
  else
  {
    v11 = a5;
    v9 = a4;
    LOWORD(v43) = 0;
    v10 = 0;
    v12 = 40LL * a4;
    v13 = 40LL * a5;
    v14 = v7 + v12;
    v15 = v7 + v13;
    v16 = *(_DWORD *)(v7 + v12 + 8);
    v17 = *(_DWORD *)(v7 + v13 + 8);
  }
  v51 = 0;
  v54 = (*(_DWORD *)v14 >> 8) & 0xFFF;
  v55 = (*(_DWORD *)v15 >> 8) & 0xFFF;
  v18 = *(_BYTE *)(v15 + 3);
  v19 = *(_BYTE *)(v15 + 4);
  v58 = ((*(_BYTE *)(v14 + 3) & 0x40) != 0) & ((*(_BYTE *)(v14 + 3) >> 4) ^ 1);
  v20 = ((v18 & 0x40) != 0) & ((v18 >> 4) ^ 1);
  v21 = *(_BYTE *)(v14 + 4);
  v59 = v20;
  v52 = v21 & 1;
  v56 = (v21 & 2) != 0;
  v57 = v19 & 1;
  v53 = (v19 & 2) != 0;
  v50 = v16 - 1;
  if ( (unsigned int)(v16 - 1) <= 0x3FFFFFFE )
  {
    v36 = a4;
    v38 = v16;
    v40 = v11;
    v41 = v9;
    v46 = v10;
    v27 = sub_2EAB300(v14);
    a4 = v36;
    v16 = v38;
    v11 = v40;
    v9 = v41;
    v51 = v27;
    v10 = v46;
  }
  v45 = 0;
  if ( (unsigned int)(v17 - 1) <= 0x3FFFFFFE )
  {
    v34 = a4;
    v35 = v16;
    v37 = v11;
    v39 = v9;
    v42 = v10;
    v26 = sub_2EAB300(v13 + *(_QWORD *)(v5 + 32));
    a4 = v34;
    v16 = v35;
    v11 = v37;
    v9 = v39;
    v45 = v26;
    v10 = v42;
  }
  if ( v6 )
  {
    if ( v10 == v16
      && (v28 = *(unsigned __int16 **)(v5 + 16), a4 < v28[1])
      && (v29 = v28[20 * *v28 + 22 + 3 * v28[8] + 3 * v9], (v29 & 1) != 0)
      && (v29 & 0xFF0) == 0 )
    {
      v59 = 0;
      v22 = v55;
      v10 = v17;
    }
    else if ( v10 == v17
           && (v30 = *(unsigned __int16 **)(v5 + 16), a5 < v30[1])
           && (v31 = v30[20 * *v30 + 22 + 3 * v30[8] + 3 * v11], (v31 & 1) != 0)
           && (v31 & 0xFF0) == 0 )
    {
      v58 = 0;
      v22 = v54;
      v10 = v16;
    }
    else
    {
      v22 = v43 & 0xFFF;
    }
    if ( a3 )
    {
      v44 = v16;
      v49 = v22;
      v63 = v10;
      v32 = (_QWORD *)sub_2E88D60(v5);
      v33 = sub_2E7B2C0(v32, v5);
      v10 = v63;
      v22 = v49;
      v16 = v44;
      v5 = (__int64)v33;
    }
    v48 = v16;
    v61 = v22;
    sub_2EAB0C0(*(_QWORD *)(v5 + 32), v10);
    **(_DWORD **)(v5 + 32) = ((v61 & 0xFFF) << 8) | **(_DWORD **)(v5 + 32) & 0xFFF000FF;
    v16 = v48;
  }
  else if ( a3 )
  {
    v62 = v16;
    v24 = (_QWORD *)sub_2E88D60(v5);
    v25 = sub_2E7B2C0(v24, v5);
    v16 = v62;
    v5 = (__int64)v25;
  }
  sub_2EAB0C0(v13 + *(_QWORD *)(v5 + 32), v16);
  sub_2EAB0C0(v12 + *(_QWORD *)(v5 + 32), v17);
  *(_DWORD *)(*(_QWORD *)(v5 + 32) + v13) = ((v54 & 0xFFF) << 8) | *(_DWORD *)(*(_QWORD *)(v5 + 32) + v13) & 0xFFF000FF;
  *(_DWORD *)(*(_QWORD *)(v5 + 32) + v12) = ((v55 & 0xFFF) << 8) | *(_DWORD *)(*(_QWORD *)(v5 + 32) + v12) & 0xFFF000FF;
  *(_BYTE *)(*(_QWORD *)(v5 + 32) + v13 + 3) = (v58 << 6) | *(_BYTE *)(*(_QWORD *)(v5 + 32) + v13 + 3) & 0xBF;
  *(_BYTE *)(*(_QWORD *)(v5 + 32) + v12 + 3) = (v59 << 6) | *(_BYTE *)(*(_QWORD *)(v5 + 32) + v12 + 3) & 0xBF;
  *(_BYTE *)(*(_QWORD *)(v5 + 32) + v13 + 4) = v52 | *(_BYTE *)(*(_QWORD *)(v5 + 32) + v13 + 4) & 0xFE;
  *(_BYTE *)(*(_QWORD *)(v5 + 32) + v12 + 4) = v57 | *(_BYTE *)(*(_QWORD *)(v5 + 32) + v12 + 4) & 0xFE;
  *(_BYTE *)(*(_QWORD *)(v5 + 32) + v13 + 4) = (2 * v56) | *(_BYTE *)(*(_QWORD *)(v5 + 32) + v13 + 4) & 0xFD;
  *(_BYTE *)(*(_QWORD *)(v5 + 32) + v12 + 4) = (2 * v53) | *(_BYTE *)(*(_QWORD *)(v5 + 32) + v12 + 4) & 0xFD;
  if ( v50 <= 0x3FFFFFFE )
    sub_2EAB350(*(_QWORD *)(v5 + 32) + v13, v51);
  if ( (unsigned int)(v17 - 1) <= 0x3FFFFFFE )
    sub_2EAB350(v12 + *(_QWORD *)(v5 + 32), v45);
  return v5;
}
