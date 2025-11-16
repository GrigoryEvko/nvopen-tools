// Function: sub_291C410
// Address: 0x291c410
//
__int64 __fastcall sub_291C410(__int64 a1, __int64 a2, int a3)
{
  __int64 v3; // r14
  __int64 v6; // rax
  __int64 **v7; // r14
  __int64 v8; // rdi
  __int64 v9; // rax
  _BYTE *v10; // r12
  __int64 v11; // rdi
  __int64 (__fastcall *v12)(__int64, unsigned int, _BYTE *, __int64); // rax
  unsigned __int8 *v13; // r15
  __int64 v14; // rax
  __int64 v15; // rdi
  unsigned __int8 *v16; // r10
  __int64 (__fastcall *v17)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8); // rax
  __int64 v18; // rax
  unsigned __int8 *v19; // r12
  __int64 v20; // rax
  __int64 v21; // rdi
  unsigned __int8 *v22; // r13
  __int64 (__fastcall *v23)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char); // rax
  __int64 v25; // r12
  __int64 v26; // rbx
  __int64 v27; // r12
  __int64 v28; // rdx
  unsigned int v29; // esi
  __int64 v30; // rdx
  __int64 v31; // r15
  __int64 v32; // rdx
  unsigned int v33; // esi
  unsigned __int8 *v34; // rax
  __int64 v35; // rax
  __int64 v36; // r12
  __int64 v37; // r12
  __int64 v38; // rdx
  unsigned int v39; // esi
  __int64 v40; // rax
  unsigned __int8 *v41; // [rsp+8h] [rbp-128h]
  __int64 v42; // [rsp+8h] [rbp-128h]
  __int64 v43; // [rsp+8h] [rbp-128h]
  unsigned __int8 *v44; // [rsp+8h] [rbp-128h]
  char *v45; // [rsp+10h] [rbp-120h] BYREF
  char v46; // [rsp+30h] [rbp-100h]
  char v47; // [rsp+31h] [rbp-FFh]
  char v48[32]; // [rsp+40h] [rbp-F0h] BYREF
  __int16 v49; // [rsp+60h] [rbp-D0h]
  char v50[32]; // [rsp+70h] [rbp-C0h] BYREF
  __int16 v51; // [rsp+90h] [rbp-A0h]
  const char *v52; // [rsp+A0h] [rbp-90h] BYREF
  char v53; // [rsp+C0h] [rbp-70h]
  char v54; // [rsp+C1h] [rbp-6Fh]
  _BYTE v55[32]; // [rsp+D0h] [rbp-60h] BYREF
  __int16 v56; // [rsp+F0h] [rbp-40h]

  v3 = a2;
  if ( a3 == 1 )
    return v3;
  v6 = sub_BCD140(**(_QWORD ***)(a2 + 8), 8 * a3);
  v54 = 1;
  v49 = 257;
  v7 = (__int64 **)v6;
  v8 = *(_QWORD *)(a2 + 8);
  v52 = "isplat";
  v53 = 3;
  v51 = 257;
  v9 = sub_AD62B0(v8);
  v10 = (_BYTE *)v9;
  if ( v7 == *(__int64 ***)(v9 + 8) )
  {
    v13 = (unsigned __int8 *)v9;
    goto LABEL_8;
  }
  v11 = *(_QWORD *)(a1 + 256);
  v12 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v11 + 120LL);
  if ( v12 != sub_920130 )
  {
    v13 = (unsigned __int8 *)v12(v11, 39u, v10, (__int64)v7);
    goto LABEL_7;
  }
  if ( *v10 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(0x27u) )
      v13 = (unsigned __int8 *)sub_ADAB70(39, (unsigned __int64)v10, v7, 0);
    else
      v13 = (unsigned __int8 *)sub_AA93C0(0x27u, (unsigned __int64)v10, (__int64)v7);
LABEL_7:
    if ( v13 )
      goto LABEL_8;
  }
  v56 = 257;
  v34 = (unsigned __int8 *)sub_BD2C40(72, 1u);
  v13 = v34;
  if ( v34 )
    sub_B515B0((__int64)v34, (__int64)v10, (__int64)v7, (__int64)v55, 0, 0);
  (*(void (__fastcall **)(_QWORD, unsigned __int8 *, char *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 264) + 16LL))(
    *(_QWORD *)(a1 + 264),
    v13,
    v48,
    *(_QWORD *)(a1 + 232),
    *(_QWORD *)(a1 + 240));
  v35 = *(_QWORD *)(a1 + 176);
  v36 = 16LL * *(unsigned int *)(a1 + 184);
  v43 = v35 + v36;
  if ( v35 != v35 + v36 )
  {
    v37 = *(_QWORD *)(a1 + 176);
    do
    {
      v38 = *(_QWORD *)(v37 + 8);
      v39 = *(_DWORD *)v37;
      v37 += 16;
      sub_B99FD0((__int64)v13, v39, v38);
    }
    while ( v43 != v37 );
  }
LABEL_8:
  v14 = sub_AD62B0((__int64)v7);
  v15 = *(_QWORD *)(a1 + 256);
  v16 = (unsigned __int8 *)v14;
  v17 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8))(*(_QWORD *)v15 + 24LL);
  if ( v17 != sub_920250 )
  {
    v44 = v16;
    v40 = v17(v15, 19u, v16, v13, 0);
    v16 = v44;
    v19 = (unsigned __int8 *)v40;
    goto LABEL_14;
  }
  if ( *v16 <= 0x15u && *v13 <= 0x15u )
  {
    v41 = v16;
    if ( (unsigned __int8)sub_AC47B0(19) )
      v18 = sub_AD5570(19, (__int64)v41, v13, 0, 0);
    else
      v18 = sub_AABE40(0x13u, v41, v13);
    v16 = v41;
    v19 = (unsigned __int8 *)v18;
LABEL_14:
    if ( v19 )
      goto LABEL_15;
  }
  v56 = 257;
  v19 = (unsigned __int8 *)sub_B504D0(19, (__int64)v16, (__int64)v13, (__int64)v55, 0, 0);
  (*(void (__fastcall **)(_QWORD, unsigned __int8 *, char *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 264) + 16LL))(
    *(_QWORD *)(a1 + 264),
    v19,
    v50,
    *(_QWORD *)(a1 + 232),
    *(_QWORD *)(a1 + 240));
  v30 = 16LL * *(unsigned int *)(a1 + 184);
  v31 = *(_QWORD *)(a1 + 176);
  v42 = v31 + v30;
  while ( v42 != v31 )
  {
    v32 = *(_QWORD *)(v31 + 8);
    v33 = *(_DWORD *)v31;
    v31 += 16;
    sub_B99FD0((__int64)v19, v33, v32);
  }
LABEL_15:
  v47 = 1;
  v45 = "zext";
  v46 = 3;
  v20 = sub_A82F30((unsigned int **)(a1 + 176), a2, (__int64)v7, (__int64)&v45, 0);
  v21 = *(_QWORD *)(a1 + 256);
  v22 = (unsigned __int8 *)v20;
  v23 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char))(*(_QWORD *)v21 + 32LL);
  if ( v23 != sub_9201A0 )
  {
    v3 = v23(v21, 17u, v22, v19, 0, 0);
    goto LABEL_20;
  }
  if ( *v22 <= 0x15u && *v19 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(17) )
      v3 = sub_AD5570(17, (__int64)v22, v19, 0, 0);
    else
      v3 = sub_AABE40(0x11u, v22, v19);
LABEL_20:
    if ( v3 )
      return v3;
  }
  v56 = 257;
  v3 = sub_B504D0(17, (__int64)v22, (__int64)v19, (__int64)v55, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, const char **, _QWORD, _QWORD))(**(_QWORD **)(a1 + 264) + 16LL))(
    *(_QWORD *)(a1 + 264),
    v3,
    &v52,
    *(_QWORD *)(a1 + 232),
    *(_QWORD *)(a1 + 240));
  v25 = 16LL * *(unsigned int *)(a1 + 184);
  v26 = *(_QWORD *)(a1 + 176);
  v27 = v26 + v25;
  while ( v27 != v26 )
  {
    v28 = *(_QWORD *)(v26 + 8);
    v29 = *(_DWORD *)v26;
    v26 += 16;
    sub_B99FD0(v3, v29, v28);
  }
  return v3;
}
