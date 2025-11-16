// Function: sub_3206EF0
// Address: 0x3206ef0
//
__int64 __fastcall sub_3206EF0(__int64 a1, __int64 a2, unsigned __int8 *a3, int a4, unsigned __int8 a5, char a6)
{
  int v7; // eax
  __int64 v8; // rcx
  unsigned __int8 v9; // al
  __int64 v10; // r13
  unsigned __int8 v11; // al
  __int64 v12; // r9
  __int64 v13; // rdx
  unsigned int v14; // r12d
  unsigned int v15; // eax
  __int64 v16; // r9
  __int64 v17; // rdx
  unsigned __int64 v18; // r8
  _DWORD *v19; // rdx
  __int64 v20; // rax
  int v21; // eax
  char v22; // cl
  int v23; // edx
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned int v26; // r12d
  __int64 v28; // rdx
  _BYTE *v29; // r12
  __int16 v30; // ax
  unsigned int *v31; // r12
  unsigned __int64 v32; // r13
  unsigned int *v33; // rax
  _DWORD *v34; // rax
  unsigned __int8 **v35; // rcx
  int v36; // eax
  unsigned int v37; // [rsp+4h] [rbp-CCh]
  unsigned int v38; // [rsp+4h] [rbp-CCh]
  int v41; // [rsp+10h] [rbp-C0h]
  int v43; // [rsp+14h] [rbp-BCh]
  int v44; // [rsp+2Ch] [rbp-A4h]
  __int16 v45; // [rsp+30h] [rbp-A0h] BYREF
  int v46; // [rsp+32h] [rbp-9Eh]
  int v47; // [rsp+36h] [rbp-9Ah]
  int v48; // [rsp+3Ah] [rbp-96h]
  char v49; // [rsp+3Eh] [rbp-92h]
  char v50; // [rsp+3Fh] [rbp-91h]
  __int16 v51; // [rsp+40h] [rbp-90h]
  int v52; // [rsp+42h] [rbp-8Eh]
  int v53; // [rsp+48h] [rbp-88h]
  __int64 v54; // [rsp+50h] [rbp-80h] BYREF
  _DWORD *v55; // [rsp+58h] [rbp-78h]
  _DWORD *v56; // [rsp+60h] [rbp-70h]
  _DWORD *v57; // [rsp+68h] [rbp-68h]
  unsigned int *v58; // [rsp+70h] [rbp-60h] BYREF
  __int64 v59; // [rsp+78h] [rbp-58h]
  _BYTE v60[80]; // [rsp+80h] [rbp-50h] BYREF

  v7 = sub_3206530(a1, a3, 0);
  v8 = a5;
  v41 = v7;
  v9 = *(_BYTE *)(a2 - 16);
  if ( (v9 & 2) != 0 )
  {
    v10 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 24LL);
    if ( v10 )
      goto LABEL_3;
LABEL_16:
    v59 = 0x800000000LL;
    v58 = (unsigned int *)v60;
    v43 = 3;
    v44 = 0;
    LODWORD(v54) = 4609;
    v55 = 0;
    goto LABEL_17;
  }
  v10 = *(_QWORD *)(a2 - 16 - 8LL * ((v9 >> 2) & 0xF) + 24);
  if ( !v10 )
    goto LABEL_16;
LABEL_3:
  LODWORD(v54) = 3;
  v58 = (unsigned int *)v60;
  v59 = 0x800000000LL;
  v11 = *(_BYTE *)(v10 - 16);
  v43 = 3;
  if ( (v11 & 2) != 0 )
  {
    v12 = *(unsigned int *)(v10 - 24);
    if ( !(_DWORD)v12 )
    {
      v44 = 0;
      if ( a5 )
        goto LABEL_12;
LABEL_28:
      v8 = 1;
      if ( *(_DWORD *)(v10 - 24) <= (unsigned int)v12 )
        goto LABEL_12;
      v28 = *(_QWORD *)(v10 - 32);
      v8 = 1;
      goto LABEL_30;
    }
    v35 = *(unsigned __int8 ***)(v10 - 32);
  }
  else
  {
    if ( (*(_WORD *)(v10 - 16) & 0x3C0) == 0 )
    {
      v44 = 0;
      v12 = 0;
      if ( a5 )
      {
        v8 = 0;
        v12 = 0;
        goto LABEL_12;
      }
      goto LABEL_48;
    }
    v35 = (unsigned __int8 **)(v10 - 16 - 8LL * ((v11 >> 2) & 0xF));
  }
  v36 = sub_3206530(a1, *v35, 0);
  v12 = 1;
  v44 = 0;
  v43 = v36;
  v8 = (*(_BYTE *)(v10 - 16) & 2) != 0;
  if ( a5 )
    goto LABEL_12;
  if ( (*(_BYTE *)(v10 - 16) & 2) != 0 )
    goto LABEL_28;
LABEL_48:
  if ( ((*(_WORD *)(v10 - 16) >> 6) & 0xFu) <= (unsigned int)v12 )
  {
    v8 = 0;
    goto LABEL_12;
  }
  v8 = 0;
  v28 = v10 - 16 - 8LL * ((*(_BYTE *)(v10 - 16) >> 2) & 0xF);
LABEL_30:
  v29 = *(_BYTE **)(v28 + 8LL * (unsigned int)v12);
  if ( v29 && *v29 == 13 )
  {
    v37 = v12;
    v30 = sub_AF18C0(*(_QWORD *)(v28 + 8LL * (unsigned int)v12));
    v12 = v37;
    if ( v30 == 15 )
    {
      v44 = sub_3206D90(a1, (__int64)v29, a2);
      v12 = v37 + 1;
    }
    v8 = (*(_BYTE *)(v10 - 16) & 2) != 0;
  }
LABEL_12:
  while ( (_BYTE)v8 )
  {
    if ( *(_DWORD *)(v10 - 24) <= (unsigned int)v12 )
      goto LABEL_35;
    v13 = *(_QWORD *)(v10 - 32);
    v14 = v12 + 1;
LABEL_9:
    v15 = sub_3206530(a1, *(unsigned __int8 **)(v13 + 8 * v12), 0);
    v17 = (unsigned int)v59;
    v18 = (unsigned int)v59 + 1LL;
    if ( v18 > HIDWORD(v59) )
    {
      v38 = v15;
      sub_C8D5F0((__int64)&v58, v60, (unsigned int)v59 + 1LL, 4u, v18, v16);
      v17 = (unsigned int)v59;
      v15 = v38;
    }
    v12 = v14;
    v58[v17] = v15;
    LODWORD(v59) = v59 + 1;
    v8 = (*(_BYTE *)(v10 - 16) & 2) != 0;
  }
  if ( ((*(_WORD *)(v10 - 16) >> 6) & 0xFu) > (unsigned int)v12 )
  {
    v14 = v12 + 1;
    v13 = v10 - 16 - 8LL * ((*(_BYTE *)(v10 - 16) >> 2) & 0xF);
    goto LABEL_9;
  }
LABEL_35:
  v31 = v58;
  v32 = (unsigned int)v59;
  if ( (_DWORD)v59 )
  {
    v33 = &v58[v32 - 1];
    LODWORD(v54) = 3;
    if ( *v33 == 3 )
    {
      *v33 = 0;
      v31 = v58;
      v32 = (unsigned int)v59;
    }
  }
  v54 = 4609;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  if ( !(v32 * 4) )
  {
LABEL_17:
    v57 = 0;
    v19 = 0;
    goto LABEL_18;
  }
  v34 = (_DWORD *)sub_22077B0(v32 * 4);
  v55 = v34;
  v57 = &v34[v32];
  if ( &v31[v32] == v31 )
  {
    v19 = v34;
  }
  else
  {
    v19 = &v34[v32];
    do
    {
      if ( v34 )
      {
        v8 = *v31;
        *v34 = v8;
      }
      ++v34;
      ++v31;
    }
    while ( v34 != v19 );
  }
LABEL_18:
  v56 = v19;
  v20 = sub_37099F0(a1 + 648, &v54, v19, v8);
  v21 = sub_3707F80(a1 + 632, v20);
  v22 = 0;
  v23 = v21;
  v24 = (unsigned int)*(unsigned __int8 *)(a2 + 44) - 177;
  if ( (unsigned int)v24 <= 0xF )
    v22 = byte_44D4F40[v24];
  v49 = v22;
  v45 = 4105;
  v46 = v43;
  v52 = v23;
  v47 = v41;
  v48 = v44;
  v50 = a6;
  v51 = v59;
  v53 = a4;
  v25 = sub_3709760(a1 + 648, &v45);
  v26 = sub_3707F80(a1 + 632, v25);
  if ( v55 )
    j_j___libc_free_0((unsigned __int64)v55);
  if ( v58 != (unsigned int *)v60 )
    _libc_free((unsigned __int64)v58);
  return v26;
}
