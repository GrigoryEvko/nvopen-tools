// Function: sub_378CBD0
// Address: 0x378cbd0
//
unsigned __int8 *__fastcall sub_378CBD0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  __int16 *v5; // rax
  __int64 v6; // rsi
  __int16 v7; // dx
  __int64 v8; // rax
  __int64 v9; // rcx
  unsigned __int16 *v10; // rdx
  int v11; // eax
  __int64 v12; // rdx
  unsigned __int16 *v13; // rdx
  __int64 v14; // r8
  unsigned __int64 v15; // rsi
  int v16; // eax
  unsigned __int16 v17; // ax
  unsigned int v18; // r15d
  int v19; // eax
  __int64 v20; // r9
  __int64 v21; // r8
  unsigned __int8 *v22; // rax
  _QWORD *v23; // rdi
  unsigned int v24; // esi
  int v25; // edx
  __int64 v26; // r9
  unsigned __int8 *v27; // rax
  _QWORD *v28; // rdi
  int v29; // edx
  __int64 v30; // r9
  unsigned __int8 *v31; // r14
  __int64 v33; // rdx
  __int64 v34; // rdx
  char v35; // [rsp+Eh] [rbp-C2h]
  char v36; // [rsp+Fh] [rbp-C1h]
  __int64 v37; // [rsp+10h] [rbp-C0h]
  __int64 *v38; // [rsp+18h] [rbp-B8h]
  __int64 v39; // [rsp+18h] [rbp-B8h]
  __int64 v40; // [rsp+40h] [rbp-90h]
  unsigned int v41; // [rsp+50h] [rbp-80h] BYREF
  __int64 v42; // [rsp+58h] [rbp-78h]
  __int128 v43; // [rsp+60h] [rbp-70h] BYREF
  __int128 v44; // [rsp+70h] [rbp-60h] BYREF
  __int64 v45; // [rsp+80h] [rbp-50h] BYREF
  int v46; // [rsp+88h] [rbp-48h]
  __int16 v47; // [rsp+90h] [rbp-40h] BYREF
  __int64 v48; // [rsp+98h] [rbp-38h]

  v5 = *(__int16 **)(a2 + 48);
  v6 = *(_QWORD *)(a2 + 80);
  v7 = *v5;
  v8 = *((_QWORD *)v5 + 1);
  *(_QWORD *)&v43 = 0;
  DWORD2(v43) = 0;
  LOWORD(v41) = v7;
  v42 = v8;
  *(_QWORD *)&v44 = 0;
  DWORD2(v44) = 0;
  v45 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v45, v6, 1);
  v46 = *(_DWORD *)(a2 + 72);
  sub_375E8D0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), (__int64)&v43, (__int64)&v44);
  v10 = (unsigned __int16 *)(*(_QWORD *)(v43 + 48) + 16LL * DWORD2(v43));
  v11 = *v10;
  v12 = *((_QWORD *)v10 + 1);
  v47 = v11;
  v48 = v12;
  if ( (_WORD)v11 )
  {
    v13 = word_4456340;
    LOBYTE(v9) = (unsigned __int16)(v11 - 176) <= 0x34u;
    v14 = (unsigned int)v9;
    v15 = word_4456340[v11 - 1];
    v16 = (unsigned __int16)v41;
    if ( (_WORD)v41 )
    {
LABEL_5:
      v37 = 0;
      v17 = word_4456580[v16 - 1];
      goto LABEL_6;
    }
  }
  else
  {
    v15 = sub_3007240((__int64)&v47);
    v16 = (unsigned __int16)v41;
    v14 = HIDWORD(v15);
    v9 = HIDWORD(v15);
    if ( (_WORD)v41 )
      goto LABEL_5;
  }
  v35 = v14;
  v36 = v9;
  v17 = sub_3009970((__int64)&v41, v15, (__int64)v13, v9, v14);
  LOBYTE(v14) = v35;
  LOBYTE(v9) = v36;
  v37 = v34;
LABEL_6:
  LODWORD(v40) = v15;
  v18 = v17;
  BYTE4(v40) = v14;
  v38 = *(__int64 **)(*(_QWORD *)(a1 + 8) + 64LL);
  if ( (_BYTE)v9 )
  {
    LOWORD(v19) = sub_2D43AD0(v17, v15);
    v21 = 0;
    if ( (_WORD)v19 )
      goto LABEL_8;
  }
  else
  {
    LOWORD(v19) = sub_2D43050(v17, v15);
    v21 = 0;
    if ( (_WORD)v19 )
      goto LABEL_8;
  }
  v19 = sub_3009450(v38, v18, v37, v40, 0, v20);
  HIWORD(v2) = HIWORD(v19);
  v21 = v33;
LABEL_8:
  LOWORD(v2) = v19;
  v39 = v21;
  v22 = sub_3406EB0(
          *(_QWORD **)(a1 + 8),
          *(_DWORD *)(a2 + 24),
          (__int64)&v45,
          v2,
          v21,
          v20,
          v43,
          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL));
  v23 = *(_QWORD **)(a1 + 8);
  v24 = *(_DWORD *)(a2 + 24);
  *(_QWORD *)&v43 = v22;
  DWORD2(v43) = v25;
  v27 = sub_3406EB0(v23, v24, (__int64)&v45, v2, v39, v26, v44, *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL));
  v28 = *(_QWORD **)(a1 + 8);
  *(_QWORD *)&v44 = v27;
  DWORD2(v44) = v29;
  v31 = sub_3406EB0(
          v28,
          0x9Fu,
          (__int64)&v45,
          v41,
          v42,
          v30,
          v43,
          __PAIR128__(*((unsigned __int64 *)&v44 + 1), (unsigned __int64)v27));
  if ( v45 )
    sub_B91220((__int64)&v45, v45);
  return v31;
}
