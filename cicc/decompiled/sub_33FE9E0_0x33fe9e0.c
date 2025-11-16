// Function: sub_33FE9E0
// Address: 0x33fe9e0
//
__int64 __fastcall sub_33FE9E0(_QWORD *a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, char a7)
{
  _QWORD *v9; // r12
  unsigned int v10; // ebx
  __int64 v11; // r13
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // r8
  void *v16; // rax
  void *v17; // rcx
  __int64 v18; // rdx
  bool v19; // si
  __int64 v20; // rcx
  __int64 v21; // rdx
  bool v22; // cl
  __int64 v23; // rcx
  void *v24; // rdx
  __int64 v25; // r8
  __int64 v26; // rbx
  __int64 v27; // rbx
  __int64 v28; // rsi
  __int64 v29; // r12
  unsigned int v30; // ebx
  __int64 result; // rax
  unsigned __int16 *v32; // rbx
  __int64 v33; // r8
  unsigned int v34; // ecx
  __int64 v35; // r13
  char v36; // al
  __int64 v37; // r13
  __int64 v38; // r13
  char v39; // al
  __int64 v40; // r13
  __int64 v41; // rdi
  bool v42; // al
  __int64 v43; // [rsp+10h] [rbp-A0h]
  __int64 v44; // [rsp+10h] [rbp-A0h]
  __int64 v45; // [rsp+18h] [rbp-98h]
  _DWORD *v46; // [rsp+18h] [rbp-98h]
  bool v47; // [rsp+18h] [rbp-98h]
  void *v48; // [rsp+20h] [rbp-90h]
  __int64 v49; // [rsp+20h] [rbp-90h]
  __int64 v50; // [rsp+20h] [rbp-90h]
  __int64 v53; // [rsp+38h] [rbp-78h]
  __int64 v54[4]; // [rsp+40h] [rbp-70h] BYREF
  __int64 v55; // [rsp+60h] [rbp-50h] BYREF
  int v56; // [rsp+68h] [rbp-48h]

  v9 = (_QWORD *)a3;
  v10 = a4;
  v11 = sub_33E1790(a3, a4, 1u, a4, a5, a6);
  v15 = sub_33E1790(a5, a6, 1u, v12, v13, v14);
  if ( !v11 )
  {
    if ( !v15 )
    {
      if ( (a7 & 0x20) == 0 )
      {
        if ( (a7 & 0x40) == 0 )
          return 0;
        goto LABEL_29;
      }
      v22 = 0;
      goto LABEL_25;
    }
    v49 = v15;
    v16 = sub_C33340();
    v15 = v49;
    v20 = *(_QWORD *)(v49 + 96);
    if ( v16 != *(void **)(v20 + 24) )
    {
      v19 = (*(_BYTE *)(v20 + 44) & 7) == 1;
      v23 = v20 + 24;
      goto LABEL_22;
    }
    goto LABEL_19;
  }
  v43 = v15;
  v45 = *(_QWORD *)(v11 + 96);
  v48 = *(void **)(v45 + 24);
  v16 = sub_C33340();
  v17 = v48;
  v18 = v45;
  v15 = v43;
  if ( v48 == v16 )
  {
    if ( (*(_BYTE *)(*(_QWORD *)(v45 + 32) + 20LL) & 7) == 1 )
    {
      v19 = 1;
      goto LABEL_16;
    }
  }
  else if ( (*(_BYTE *)(v45 + 44) & 7) == 1 )
  {
    v19 = 1;
    goto LABEL_9;
  }
  v19 = 0;
  if ( v43 )
  {
    v20 = *(_QWORD *)(v43 + 96);
    if ( v16 != *(void **)(v20 + 24) )
    {
      v19 = (*(_BYTE *)(v20 + 44) & 7) == 1;
LABEL_7:
      v17 = *(void **)(v18 + 24);
      goto LABEL_8;
    }
LABEL_19:
    v19 = (*(_BYTE *)(*(_QWORD *)(v20 + 32) + 20LL) & 7) == 1;
    if ( !v11 )
    {
LABEL_21:
      v23 = *(_QWORD *)(v20 + 32);
      goto LABEL_22;
    }
    v18 = *(_QWORD *)(v11 + 96);
    goto LABEL_7;
  }
LABEL_8:
  if ( v17 == v16 )
  {
LABEL_16:
    v21 = *(_QWORD *)(v18 + 32);
    goto LABEL_10;
  }
LABEL_9:
  v21 = v18 + 24;
LABEL_10:
  v22 = 1;
  if ( (*(_BYTE *)(v21 + 20) & 7) == 0 )
    goto LABEL_23;
  v22 = 0;
  if ( !v15 )
    goto LABEL_23;
  v20 = *(_QWORD *)(v15 + 96);
  if ( v16 == *(void **)(v20 + 24) )
    goto LABEL_21;
  v23 = v20 + 24;
LABEL_22:
  v22 = (*(_BYTE *)(v23 + 20) & 7) == 0;
LABEL_23:
  if ( (a7 & 0x20) == 0 )
    goto LABEL_27;
  if ( v19 )
    goto LABEL_47;
LABEL_25:
  if ( *((_DWORD *)v9 + 6) == 51 || *(_DWORD *)(a5 + 24) == 51 )
    goto LABEL_47;
LABEL_27:
  if ( (a7 & 0x40) == 0 )
    goto LABEL_31;
  if ( v22 )
  {
LABEL_47:
    v32 = (unsigned __int16 *)(v9[6] + 16LL * v10);
    v33 = *((_QWORD *)v32 + 1);
    v34 = *v32;
    v55 = 0;
    v56 = 0;
    v9 = sub_33F17F0(a1, 51, (__int64)&v55, v34, v33);
    if ( v55 )
      sub_B91220((__int64)&v55, v55);
    return (__int64)v9;
  }
LABEL_29:
  if ( *((_DWORD *)v9 + 6) == 51 || *(_DWORD *)(a5 + 24) == 51 )
    goto LABEL_47;
LABEL_31:
  if ( !v15 )
    return 0;
  if ( a2 == 96 )
  {
    v35 = *(_QWORD *)(v15 + 96);
    if ( *(void **)(v35 + 24) == sub_C33340() )
    {
      v37 = *(_QWORD *)(v35 + 32);
      if ( (*(_BYTE *)(v37 + 20) & 7) != 3 )
        return 0;
    }
    else
    {
      v36 = *(_BYTE *)(v35 + 44);
      v37 = v35 + 24;
      if ( (v36 & 7) != 3 )
        return 0;
    }
    if ( (*(_BYTE *)(v37 + 20) & 8) != 0 )
      return (__int64)v9;
    return 0;
  }
  if ( a2 == 97 )
  {
    v38 = *(_QWORD *)(v15 + 96);
    if ( *(void **)(v38 + 24) == sub_C33340() )
    {
      v40 = *(_QWORD *)(v38 + 32);
      if ( (*(_BYTE *)(v40 + 20) & 7) != 3 )
        return 0;
    }
    else
    {
      v39 = *(_BYTE *)(v38 + 44);
      v40 = v38 + 24;
      if ( (v39 & 7) != 3 )
        return 0;
    }
    if ( (*(_BYTE *)(v40 + 20) & 8) == 0 )
      return (__int64)v9;
    return 0;
  }
  if ( (unsigned int)(a2 - 98) > 1 )
    return 0;
  v44 = v15;
  v50 = *(_QWORD *)(v15 + 96);
  v46 = sub_C33320();
  sub_C3B1B0((__int64)&v55, 1.0);
  sub_C407B0(v54, &v55, v46);
  sub_C338F0((__int64)&v55);
  sub_C41640(v54, *(_DWORD **)(v50 + 24), 1, (bool *)&v55);
  v24 = *(void **)(v50 + 24);
  if ( v24 != (void *)v54[0] )
  {
    sub_91D830(v54);
    v25 = v44;
    goto LABEL_37;
  }
  v41 = v50 + 24;
  if ( v24 == sub_C33340() )
    v42 = sub_C3E590(v41, (__int64)v54);
  else
    v42 = sub_C33D00(v41, (__int64)v54);
  v47 = v42;
  sub_91D830(v54);
  v25 = v44;
  if ( v47 )
    return (__int64)v9;
LABEL_37:
  if ( (a7 & 0x20) == 0 || a2 != 98 || a7 >= 0 )
    return 0;
  v26 = *(_QWORD *)(v25 + 96);
  v27 = *(void **)(v26 + 24) == sub_C33340() ? *(_QWORD *)(v26 + 32) : v26 + 24;
  if ( (*(_BYTE *)(v27 + 20) & 7) != 3 )
    return 0;
  v28 = *(_QWORD *)(a5 + 80);
  v29 = *(_QWORD *)(*(_QWORD *)(a5 + 48) + 16LL * (unsigned int)a6 + 8);
  v30 = *(unsigned __int16 *)(*(_QWORD *)(a5 + 48) + 16LL * (unsigned int)a6);
  v55 = v28;
  if ( v28 )
    sub_B96E90((__int64)&v55, v28, 1);
  v56 = *(_DWORD *)(a5 + 72);
  result = sub_33FE730((__int64)a1, (__int64)&v55, v30, v29, 0, (__m128i)0LL);
  if ( v55 )
  {
    v53 = result;
    sub_B91220((__int64)&v55, v55);
    return v53;
  }
  return result;
}
