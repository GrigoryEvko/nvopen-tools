// Function: sub_258D9E0
// Address: 0x258d9e0
//
__int64 __fastcall sub_258D9E0(__int64 a1, __int64 a2)
{
  unsigned __int8 **v3; // rdx
  unsigned __int8 v4; // al
  __int64 *v5; // r14
  unsigned __int8 v6; // r13
  char v7; // bl
  __int64 v8; // r12
  __int64 v9; // r15
  __m128i v10; // rax
  __int64 v11; // rsi
  __int64 v13; // rax
  __int64 *v14; // r12
  __int64 v15; // r15
  char v16; // r13
  _BYTE *v17; // rdx
  __int64 v18; // rdi
  unsigned __int8 *v19; // r8
  __int64 *v20; // rdx
  unsigned __int64 v21; // rcx
  __int64 v22; // r9
  __int64 (__fastcall *v23)(__int64); // rax
  __int64 v24; // rax
  __int64 *v25; // r12
  int v26; // eax
  unsigned __int8 v27; // dl
  __int64 v28; // rdi
  __int64 v29; // r13
  __int64 (*v30)(void); // rax
  _BYTE *v31; // rdx
  __int64 v32; // [rsp-8h] [rbp-98h]
  unsigned __int64 v33; // [rsp+0h] [rbp-90h]
  unsigned __int8 *v34; // [rsp+8h] [rbp-88h]
  __int64 v35; // [rsp+10h] [rbp-80h]
  bool v36; // [rsp+1Eh] [rbp-72h]
  char v37; // [rsp+1Fh] [rbp-71h]
  unsigned __int8 *v38; // [rsp+28h] [rbp-68h]
  _BYTE *v39; // [rsp+30h] [rbp-60h]
  __int64 v40; // [rsp+38h] [rbp-58h]
  unsigned __int8 v41; // [rsp+38h] [rbp-58h]
  unsigned __int8 *i; // [rsp+40h] [rbp-50h]
  __m128i v44; // [rsp+50h] [rbp-40h] BYREF

  if ( (*(_BYTE *)(a2 + 7) & 0x40) == 0 )
  {
    v26 = *(_DWORD *)(a2 + 4);
    v27 = **(_BYTE **)(a2 - 32LL * (v26 & 0x7FFFFFF));
    v38 = *(unsigned __int8 **)(a2 - 32LL * (v26 & 0x7FFFFFF));
    if ( v27 <= 0x1Cu )
      goto LABEL_3;
    if ( (v27 & 0xFD) != 0x54 )
    {
      v37 = 1;
      v5 = *(__int64 **)(a1 + 24);
LABEL_31:
      v3 = (unsigned __int8 **)(a2 - 32LL * (v26 & 0x7FFFFFF));
LABEL_32:
      v38 = *v3;
      goto LABEL_4;
    }
LABEL_27:
    v28 = *(_QWORD *)(a1 + 8);
    v29 = **(_QWORD **)(a1 + 16);
    v30 = *(__int64 (**)(void))(*(_QWORD *)v28 + 40LL);
    if ( (char *)v30 == (char *)sub_2505DE0 )
    {
      v31 = (_BYTE *)(v28 + 88);
    }
    else
    {
      v28 = *(_QWORD *)(a1 + 8);
      v31 = (_BYTE *)v30();
    }
    sub_258BA20(v28, *(_QWORD *)a1, v31, (unsigned __int64)v38, (unsigned __int8 *)a2, 3, v29);
    v37 = 0;
    v5 = *(__int64 **)(a1 + 24);
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    {
      v3 = *(unsigned __int8 ***)(a2 - 8);
      goto LABEL_32;
    }
    v26 = *(_DWORD *)(a2 + 4);
    goto LABEL_31;
  }
  v3 = *(unsigned __int8 ***)(a2 - 8);
  v38 = *v3;
  v4 = **v3;
  if ( v4 > 0x1Cu )
  {
    if ( (v4 & 0xFD) != 0x54 )
    {
      v37 = 1;
      v5 = *(__int64 **)(a1 + 24);
      goto LABEL_32;
    }
    goto LABEL_27;
  }
LABEL_3:
  v37 = 1;
  v5 = *(__int64 **)(a1 + 24);
LABEL_4:
  v6 = 2;
  for ( i = (unsigned __int8 *)&unk_438A62B; ; v6 = *i )
  {
    v7 = v6;
    *(_DWORD *)(*v5 + 8) = 0;
    v8 = v5[1];
    v39 = (_BYTE *)v5[3];
    v9 = v5[2];
    v40 = *v5;
    v10.m128i_i64[0] = sub_250D2C0((unsigned __int64)v38, 0);
    v44 = v10;
    v11 = v32;
    v41 = sub_2526B50(v8, &v44, v9, v40, v6, v39, 1u);
    if ( !v41 )
      break;
    if ( !v37 )
      goto LABEL_7;
    v13 = *v5;
    if ( v6 == 2 )
    {
      v25 = (__int64 *)(*(_QWORD *)v13 + 16LL * *(unsigned int *)(v13 + 8));
      v11 = (__int64)v25;
      v36 = v25 == sub_2537880(*(__int64 **)v13, (__int64)v25, (__int64 *)v5[4]);
      v14 = *(__int64 **)*v5;
      v15 = (__int64)&v14[2 * *(unsigned int *)(*v5 + 8)];
      if ( v14 == (__int64 *)v15 )
        goto LABEL_23;
    }
    else
    {
      v14 = *(__int64 **)v13;
      v15 = *(_QWORD *)v13 + 16LL * *(unsigned int *)(v13 + 8);
      if ( v15 == *(_QWORD *)v13 )
        goto LABEL_7;
      v36 = 0;
    }
    v16 = 3;
    if ( !v36 )
      v16 = v7;
    do
    {
      v18 = v5[2];
      v19 = (unsigned __int8 *)v14[1];
      v21 = *v14;
      if ( !v19 )
        v19 = (unsigned __int8 *)a2;
      v20 = (__int64 *)v5[4];
      v22 = *v20;
      v23 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v18 + 40LL);
      if ( v23 == sub_2505DE0 )
      {
        v17 = (_BYTE *)(v18 + 88);
      }
      else
      {
        v33 = *v14;
        v34 = v19;
        v35 = *v20;
        v24 = ((__int64 (__fastcall *)(__int64, __int64))v23)(v18, v11);
        v21 = v33;
        v19 = v34;
        v22 = v35;
        v17 = (_BYTE *)v24;
      }
      v11 = v5[1];
      v14 += 2;
      sub_258BA20(v18, v11, v17, v21, v19, v16, v22);
    }
    while ( (__int64 *)v15 != v14 );
LABEL_23:
    if ( v36 )
      return v41;
LABEL_7:
    if ( &unk_438A62D == (_UNKNOWN *)++i )
      return v41;
  }
  return v41;
}
