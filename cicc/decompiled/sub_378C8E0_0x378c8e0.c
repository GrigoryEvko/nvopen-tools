// Function: sub_378C8E0
// Address: 0x378c8e0
//
unsigned __int8 *__fastcall sub_378C8E0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  __int64 v4; // rsi
  __int64 *v5; // rax
  int v6; // eax
  unsigned __int64 *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r8
  unsigned __int16 *v10; // rax
  int v11; // r14d
  unsigned __int16 *v12; // rdx
  int v13; // eax
  __int64 v14; // rdx
  unsigned __int16 *v15; // rdx
  __int64 v16; // rcx
  unsigned __int64 v17; // rsi
  unsigned __int16 v18; // ax
  unsigned int v19; // r14d
  int v20; // eax
  __int64 v21; // r9
  __int64 v22; // r8
  __int128 v23; // rax
  __int64 v24; // r9
  __int128 v25; // rax
  __int64 v26; // r9
  unsigned __int8 *v27; // r14
  __int64 v29; // rdx
  __int64 v30; // rdx
  char v31; // [rsp+6h] [rbp-CAh]
  char v32; // [rsp+7h] [rbp-C9h]
  __int64 v33; // [rsp+8h] [rbp-C8h]
  __int64 v34; // [rsp+8h] [rbp-C8h]
  __int64 *v35; // [rsp+10h] [rbp-C0h]
  __int128 v36; // [rsp+10h] [rbp-C0h]
  __int64 v37; // [rsp+30h] [rbp-A0h] BYREF
  int v38; // [rsp+38h] [rbp-98h]
  __int128 v39; // [rsp+40h] [rbp-90h] BYREF
  __int128 v40; // [rsp+50h] [rbp-80h] BYREF
  __int128 v41; // [rsp+60h] [rbp-70h] BYREF
  __int128 v42; // [rsp+70h] [rbp-60h] BYREF
  unsigned int v43; // [rsp+80h] [rbp-50h] BYREF
  __int64 v44; // [rsp+88h] [rbp-48h]
  __int64 v45[8]; // [rsp+90h] [rbp-40h] BYREF

  v4 = *(_QWORD *)(a2 + 80);
  v5 = *(__int64 **)(*(_QWORD *)(a1 + 8) + 64LL);
  v37 = v4;
  v35 = v5;
  if ( v4 )
    sub_B96E90((__int64)&v37, v4, 1);
  v6 = *(_DWORD *)(a2 + 72);
  DWORD2(v40) = 0;
  DWORD2(v39) = 0;
  v38 = v6;
  v7 = *(unsigned __int64 **)(a2 + 40);
  DWORD2(v41) = 0;
  DWORD2(v42) = 0;
  v8 = v7[1];
  *(_QWORD *)&v39 = 0;
  *(_QWORD *)&v40 = 0;
  *(_QWORD *)&v41 = 0;
  *(_QWORD *)&v42 = 0;
  sub_375E8D0(a1, *v7, v8, (__int64)&v39, (__int64)&v40);
  sub_375E8D0(
    a1,
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
    (__int64)&v41,
    (__int64)&v42);
  v10 = *(unsigned __int16 **)(a2 + 48);
  v11 = *v10;
  v44 = *((_QWORD *)v10 + 1);
  LOWORD(v43) = v11;
  v12 = (unsigned __int16 *)(*(_QWORD *)(v39 + 48) + 16LL * DWORD2(v39));
  v13 = *v12;
  v14 = *((_QWORD *)v12 + 1);
  LOWORD(v45[0]) = v13;
  v45[1] = v14;
  if ( (_WORD)v13 )
  {
    v15 = word_4456340;
    LOBYTE(v9) = (unsigned __int16)(v13 - 176) <= 0x34u;
    v16 = (unsigned int)v9;
    v17 = word_4456340[v13 - 1];
    if ( (_WORD)v11 )
    {
LABEL_5:
      v33 = 0;
      v18 = word_4456580[v11 - 1];
      goto LABEL_6;
    }
  }
  else
  {
    v17 = sub_3007240((__int64)v45);
    v16 = HIDWORD(v17);
    v9 = HIDWORD(v17);
    if ( (_WORD)v11 )
      goto LABEL_5;
  }
  v31 = v9;
  v32 = v16;
  v18 = sub_3009970((__int64)&v43, v17, (__int64)v15, v16, v9);
  LOBYTE(v9) = v31;
  v33 = v30;
  LOBYTE(v16) = v32;
LABEL_6:
  LODWORD(v45[0]) = v17;
  v19 = v18;
  BYTE4(v45[0]) = v16;
  if ( (_BYTE)v9 )
  {
    LOWORD(v20) = sub_2D43AD0(v18, v17);
    v22 = 0;
    if ( (_WORD)v20 )
      goto LABEL_8;
  }
  else
  {
    LOWORD(v20) = sub_2D43050(v18, v17);
    v22 = 0;
    if ( (_WORD)v20 )
      goto LABEL_8;
  }
  v20 = sub_3009450(v35, v19, v33, v45[0], 0, v21);
  HIWORD(v2) = HIWORD(v20);
  v22 = v29;
LABEL_8:
  LOWORD(v2) = v20;
  v34 = v22;
  *(_QWORD *)&v23 = sub_3406EB0(*(_QWORD **)(a1 + 8), *(_DWORD *)(a2 + 24), (__int64)&v37, v2, v22, v21, v39, v41);
  v36 = v23;
  *(_QWORD *)&v25 = sub_3406EB0(*(_QWORD **)(a1 + 8), *(_DWORD *)(a2 + 24), (__int64)&v37, v2, v34, v24, v40, v42);
  v27 = sub_3406EB0(*(_QWORD **)(a1 + 8), 0x9Fu, (__int64)&v37, v43, v44, v26, v36, v25);
  if ( v37 )
    sub_B91220((__int64)&v37, v37);
  return v27;
}
