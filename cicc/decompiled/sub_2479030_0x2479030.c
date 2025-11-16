// Function: sub_2479030
// Address: 0x2479030
//
void __fastcall sub_2479030(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rbx
  _BYTE *v6; // rax
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  _BYTE *v10; // rdx
  _BYTE *v11; // rbx
  unsigned int v12; // eax
  __int64 v13; // rax
  __int16 v14; // di
  unsigned __int8 *v15; // rbx
  unsigned int v16; // eax
  __int64 v17; // rax
  unsigned __int8 *v18; // r10
  __int64 (__fastcall *v19)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v20; // rax
  __int64 v21; // r15
  unsigned int *v22; // rbx
  unsigned int *v23; // r14
  __int64 v24; // rdx
  unsigned int v25; // esi
  __int64 v26; // rax
  __int64 v27; // [rsp+0h] [rbp-160h]
  _BYTE *v28; // [rsp+0h] [rbp-160h]
  __int64 v29; // [rsp+8h] [rbp-158h]
  __int64 v30; // [rsp+8h] [rbp-158h]
  __int64 v31; // [rsp+10h] [rbp-150h]
  __int64 v32; // [rsp+10h] [rbp-150h]
  __int64 v33; // [rsp+10h] [rbp-150h]
  __int64 v34; // [rsp+18h] [rbp-148h]
  __int64 v35; // [rsp+18h] [rbp-148h]
  _BYTE *v36; // [rsp+18h] [rbp-148h]
  unsigned __int8 *v37; // [rsp+18h] [rbp-148h]
  unsigned __int8 *v38; // [rsp+18h] [rbp-148h]
  bool v39; // [rsp+2Fh] [rbp-131h] BYREF
  _QWORD v40[2]; // [rsp+30h] [rbp-130h] BYREF
  char v41[32]; // [rsp+40h] [rbp-120h] BYREF
  __int16 v42; // [rsp+60h] [rbp-100h]
  _BYTE v43[32]; // [rsp+70h] [rbp-F0h] BYREF
  __int16 v44; // [rsp+90h] [rbp-D0h]
  unsigned int *v45; // [rsp+A0h] [rbp-C0h] BYREF
  int v46; // [rsp+A8h] [rbp-B8h]
  char v47; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v48; // [rsp+D8h] [rbp-88h]
  __int64 v49; // [rsp+E0h] [rbp-80h]
  __int64 v50; // [rsp+F0h] [rbp-70h]
  __int64 v51; // [rsp+F8h] [rbp-68h]
  void *v52; // [rsp+120h] [rbp-40h]

  sub_23D0AB0((__int64)&v45, a2, 0, 0, 0);
  v27 = *(_QWORD *)(a2 - 64);
  v31 = *(_QWORD *)(a2 - 32);
  v34 = sub_246F3F0(a1, v27);
  v29 = v31;
  v4 = sub_246F3F0(a1, v31);
  v44 = 257;
  v5 = v4;
  v32 = v34;
  v6 = sub_94BCF0(&v45, v27, *(_QWORD *)(v34 + 8), (__int64)v43);
  v44 = 257;
  v35 = (__int64)v6;
  v28 = sub_94BCF0(&v45, v29, *(_QWORD *)(v5 + 8), (__int64)v43);
  v39 = sub_B532B0(*(_WORD *)(a2 + 2) & 0x3F);
  v40[0] = &v39;
  v40[1] = &v45;
  v7 = sub_2463BA0((__int64)v40, v35, v32);
  v33 = v8;
  v30 = v7;
  v9 = sub_2463BA0((__int64)v40, (__int64)v28, v5);
  v44 = 257;
  v11 = v10;
  v36 = (_BYTE *)v9;
  v12 = sub_B52EF0(*(_WORD *)(a2 + 2) & 0x3F);
  v13 = sub_92B530(&v45, v12, v30, v11, (__int64)v43);
  v14 = *(_WORD *)(a2 + 2);
  v44 = 257;
  v15 = (unsigned __int8 *)v13;
  v16 = sub_B52EF0(v14 & 0x3F);
  v17 = sub_92B530(&v45, v16, v33, v36, (__int64)v43);
  v42 = 257;
  v18 = (unsigned __int8 *)v17;
  v19 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))(*(_QWORD *)v50 + 16LL);
  if ( v19 != sub_9202E0 )
  {
    v38 = v18;
    v26 = v19(v50, 30u, v15, v18);
    v18 = v38;
    v21 = v26;
    goto LABEL_7;
  }
  if ( *v15 <= 0x15u && *v18 <= 0x15u )
  {
    v37 = v18;
    if ( (unsigned __int8)sub_AC47B0(30) )
      v20 = sub_AD5570(30, (__int64)v15, v37, 0, 0);
    else
      v20 = sub_AABE40(0x1Eu, v15, v37);
    v18 = v37;
    v21 = v20;
LABEL_7:
    if ( v21 )
      goto LABEL_8;
  }
  v44 = 257;
  v21 = sub_B504D0(30, (__int64)v15, (__int64)v18, (__int64)v43, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, char *, __int64, __int64))(*(_QWORD *)v51 + 16LL))(v51, v21, v41, v48, v49);
  v22 = v45;
  v23 = &v45[4 * v46];
  if ( v45 != v23 )
  {
    do
    {
      v24 = *((_QWORD *)v22 + 1);
      v25 = *v22;
      v22 += 4;
      sub_B99FD0(v21, v25, v24);
    }
    while ( v23 != v22 );
  }
LABEL_8:
  sub_246EF60(a1, a2, v21);
  if ( *(_DWORD *)(*(_QWORD *)(a1 + 8) + 4LL) )
    sub_2477350(a1, a2);
  nullsub_61();
  v52 = &unk_49DA100;
  nullsub_63();
  if ( v45 != (unsigned int *)&v47 )
    _libc_free((unsigned __int64)v45);
}
