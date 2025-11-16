// Function: sub_938F90
// Address: 0x938f90
//
__int64 __fastcall sub_938F90(__int64 a1, __int64 a2, _DWORD *a3, __int64 a4)
{
  __int64 v4; // r14
  unsigned __int64 v6; // rbx
  __int64 v7; // rax
  unsigned __int64 v9; // r15
  unsigned int v10; // eax
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // r13
  __int64 result; // rax
  unsigned int *v15; // rbx
  unsigned int *i; // r12
  __int64 v17; // rdx
  __int64 v18; // rsi
  __int64 v19; // rdi
  int v20; // r8d
  __int64 v21; // rdi
  unsigned __int64 v22; // rcx
  int v23; // eax
  int v24; // r8d
  __int64 v25; // r13
  int v26; // r9d
  __int64 v27; // rax
  __int64 v28; // r15
  unsigned int *v29; // rbx
  unsigned int *v30; // r13
  __int64 v31; // rdx
  __int64 v32; // rsi
  __int64 v33; // r13
  __int64 v34; // rax
  __int64 v35; // rbx
  __int64 v36; // r13
  unsigned int *v37; // r12
  unsigned int *v38; // r13
  __int64 v39; // rdx
  __int64 v40; // rsi
  __int64 v41; // rax
  unsigned __int8 v42; // al
  bool v43; // al
  int v44; // [rsp+0h] [rbp-A0h]
  int v45; // [rsp+4h] [rbp-9Ch]
  int v46; // [rsp+4h] [rbp-9Ch]
  char v47; // [rsp+8h] [rbp-98h]
  int v48; // [rsp+8h] [rbp-98h]
  _BYTE v49[32]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v50; // [rsp+30h] [rbp-70h]
  _BYTE v51[32]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v52; // [rsp+60h] [rbp-40h]

  v4 = a1 + 48;
  v6 = *(_QWORD *)(a1 + 208);
  if ( !v6 )
    goto LABEL_5;
  v7 = *(_QWORD *)(a2 + 16);
  v9 = *(_QWORD *)(v7 + 24);
  v10 = *(_DWORD *)(v7 + 12);
  if ( v10 == 2 )
  {
    if ( !sub_91B770(v9) )
      sub_91B8A0("Indirect returns for non-aggregate values not supported!", a3, 1);
    goto LABEL_5;
  }
  if ( v10 > 2 )
  {
    if ( v10 != 3 )
      sub_91B8A0("Unsupported ABI variant!", a3, 1);
    goto LABEL_5;
  }
  v19 = *(_QWORD *)(a1 + 32);
  v50 = 257;
  v20 = unk_4D0463C;
  if ( unk_4D0463C )
  {
    v43 = sub_90AA40(v19, v6);
    v6 = *(_QWORD *)(a1 + 208);
    v19 = *(_QWORD *)(a1 + 32);
    v20 = v43;
  }
  v21 = v19 + 8;
  if ( *(_DWORD *)(a1 + 216) )
  {
    _BitScanReverse64(&v22, *(unsigned int *)(a1 + 216));
    v45 = v20;
    v47 = v22 ^ 0x3F;
    v23 = sub_91A390(v21, v9, 0, v22 ^ 0x3F);
    v24 = v45;
    LODWORD(v25) = v23;
    v26 = (unsigned __int8)(63 - v47);
  }
  else
  {
    v48 = v20;
    v25 = sub_91A390(v21, v9, 0, a4);
    v41 = sub_AA4E30(*(_QWORD *)(a1 + 96));
    v42 = sub_AE5020(v41, v25);
    v24 = v48;
    v26 = v42;
  }
  v44 = v26;
  v52 = 257;
  v46 = v24;
  v27 = sub_BD2C40(80, unk_3F10A14);
  v28 = v27;
  if ( !v27 )
  {
    (*(void (__fastcall **)(_QWORD, _QWORD, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 136) + 16LL))(
      *(_QWORD *)(a1 + 136),
      0,
      v49,
      *(_QWORD *)(v4 + 56),
      *(_QWORD *)(v4 + 64));
    v29 = *(unsigned int **)(a1 + 48);
    v30 = &v29[4 * *(unsigned int *)(a1 + 56)];
    if ( v30 == v29 )
      goto LABEL_5;
    goto LABEL_16;
  }
  sub_B4D190(v27, v25, v6, (unsigned int)v51, v46, v44, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 136) + 16LL))(
    *(_QWORD *)(a1 + 136),
    v28,
    v49,
    *(_QWORD *)(v4 + 56),
    *(_QWORD *)(v4 + 64));
  v29 = *(unsigned int **)(a1 + 48);
  v30 = &v29[4 * *(unsigned int *)(a1 + 56)];
  if ( v29 != v30 )
  {
    do
    {
LABEL_16:
      v31 = *((_QWORD *)v29 + 1);
      v32 = *v29;
      v29 += 4;
      sub_B99FD0(v28, v32, v31);
    }
    while ( v30 != v29 );
    if ( v28 )
      goto LABEL_18;
LABEL_5:
    v11 = *(_QWORD *)(a1 + 120);
    v52 = 257;
    v12 = sub_BD2C40(72, 0);
    v13 = v12;
    if ( v12 )
      sub_B4BB80(v12, v11, 0, 0, 0, 0);
    result = (*(__int64 (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 136) + 16LL))(
               *(_QWORD *)(a1 + 136),
               v13,
               v51,
               *(_QWORD *)(v4 + 56),
               *(_QWORD *)(v4 + 64));
    v15 = *(unsigned int **)(a1 + 48);
    for ( i = &v15[4 * *(unsigned int *)(a1 + 56)]; i != v15; result = sub_B99FD0(v13, v18, v17) )
    {
      v17 = *((_QWORD *)v15 + 1);
      v18 = *v15;
      v15 += 4;
    }
    return result;
  }
LABEL_18:
  v33 = *(_QWORD *)(a1 + 120);
  v52 = 257;
  v34 = sub_BD2C40(72, 1);
  v35 = v34;
  if ( v34 )
    sub_B4BB80(v34, v33, v28, 1, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 136) + 16LL))(
    *(_QWORD *)(a1 + 136),
    v35,
    v51,
    *(_QWORD *)(v4 + 56),
    *(_QWORD *)(v4 + 64));
  result = *(_QWORD *)(a1 + 48);
  v36 = 16LL * *(unsigned int *)(a1 + 56);
  v37 = (unsigned int *)result;
  v38 = (unsigned int *)(result + v36);
  if ( (unsigned int *)result != v38 )
  {
    do
    {
      v39 = *((_QWORD *)v37 + 1);
      v40 = *v37;
      v37 += 4;
      result = sub_B99FD0(v35, v40, v39);
    }
    while ( v38 != v37 );
  }
  return result;
}
