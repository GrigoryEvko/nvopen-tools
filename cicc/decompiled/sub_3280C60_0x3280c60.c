// Function: sub_3280C60
// Address: 0x3280c60
//
__int64 __fastcall sub_3280C60(_QWORD *a1, __int64 a2)
{
  __int64 *v4; // rax
  int v5; // r15d
  __int64 v6; // r8
  int v7; // ecx
  __int64 v8; // r12
  __int64 v9; // r13
  __int16 v10; // dx
  __int64 v11; // rsi
  int v12; // eax
  __int64 result; // rax
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rax
  int v17; // esi
  __int64 v18; // rax
  _DWORD *v19; // rax
  __int64 v20; // rsi
  __int64 v21; // r13
  __int64 v22; // rax
  unsigned __int16 v23; // r12
  bool v24; // al
  __int64 v25; // rdx
  __int64 v26; // rsi
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rdi
  unsigned __int16 *v30; // rdx
  int v31; // r9d
  __int128 v32; // rax
  int v33; // r9d
  __int64 v34; // rdx
  _DWORD *v35; // rax
  __int64 v36; // r15
  int v37; // r9d
  __int64 v38; // rcx
  __int64 v39; // rdi
  unsigned __int16 *v40; // rax
  __int128 v41; // rax
  int v42; // r9d
  __int128 v43; // [rsp-10h] [rbp-80h]
  __int128 v44; // [rsp-10h] [rbp-80h]
  __int64 v45; // [rsp+8h] [rbp-68h]
  __int64 v46; // [rsp+8h] [rbp-68h]
  __int64 v47; // [rsp+10h] [rbp-60h]
  unsigned int v48; // [rsp+10h] [rbp-60h]
  unsigned int v49; // [rsp+10h] [rbp-60h]
  __int64 v50; // [rsp+18h] [rbp-58h]
  __int64 v51; // [rsp+20h] [rbp-50h] BYREF
  __int64 v52; // [rsp+28h] [rbp-48h]
  __int64 v53; // [rsp+30h] [rbp-40h] BYREF
  int v54; // [rsp+38h] [rbp-38h]

  v4 = *(__int64 **)(a2 + 40);
  v5 = *(_DWORD *)(a2 + 24);
  v6 = *v4;
  v7 = *((_DWORD *)v4 + 2);
  v8 = v4[5];
  v9 = v4[6];
  v10 = *(_WORD *)(v8 + 96);
  v11 = *(_QWORD *)(v8 + 104);
  v12 = *(_DWORD *)(*v4 + 24);
  LOWORD(v51) = v10;
  v52 = v11;
  if ( v5 == v12 )
  {
    v14 = *(_QWORD *)(*(_QWORD *)(v6 + 40) + 40LL);
    if ( *(_WORD *)(v14 + 96) == v10 && (*(_QWORD *)(v14 + 104) == v11 || v10) )
      return v6;
  }
  if ( v12 != 216 )
    return 0;
  v15 = *(_QWORD *)(v6 + 56);
  if ( !v15 )
    return 0;
  v16 = *(_QWORD *)(v6 + 56);
  v17 = 1;
  while ( 1 )
  {
    while ( *(_DWORD *)(v16 + 8) != v7 )
    {
      v16 = *(_QWORD *)(v16 + 32);
      if ( !v16 )
        goto LABEL_17;
    }
    if ( !v17 )
      break;
    v18 = *(_QWORD *)(v16 + 32);
    if ( !v18 )
      goto LABEL_18;
    if ( *(_DWORD *)(v18 + 8) == v7 )
      goto LABEL_29;
    v16 = *(_QWORD *)(v18 + 32);
    v17 = 0;
    if ( !v16 )
    {
LABEL_17:
      if ( v17 == 1 )
        goto LABEL_29;
LABEL_18:
      v19 = *(_DWORD **)(v6 + 40);
      v17 = 1;
      if ( v5 != *(_DWORD *)(*(_QWORD *)v19 + 24LL) )
        goto LABEL_29;
      v20 = *(_QWORD *)(a2 + 80);
      v47 = v6;
      v53 = v20;
      if ( v20 )
      {
        sub_B96E90((__int64)&v53, v20, 1);
        v19 = *(_DWORD **)(v47 + 40);
      }
      v54 = *(_DWORD *)(a2 + 72);
      v21 = *(_QWORD *)v19;
      v48 = v19[2];
      v22 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v19 + 40LL) + 40LL);
      v23 = *(_WORD *)(v22 + 96);
      v45 = *(_QWORD *)(v22 + 104);
      v24 = sub_3280B30((__int64)&v51, v23, v45);
      v25 = v45;
      if ( v24 )
      {
        v26 = v51;
        v25 = v52;
      }
      else
      {
        v26 = v23;
      }
      v27 = sub_33F7D60(*a1, v26, v25);
      v29 = v28;
      v30 = (unsigned __int16 *)(*(_QWORD *)(v21 + 48) + 16LL * v48);
      *((_QWORD *)&v43 + 1) = v29;
      *(_QWORD *)&v43 = v27;
      *(_QWORD *)&v32 = sub_3406EB0(
                          *a1,
                          v5,
                          (unsigned int)&v53,
                          *v30,
                          *((_QWORD *)v30 + 1),
                          v31,
                          *(_OWORD *)*(_QWORD *)(v21 + 40),
                          v43);
      result = sub_33FAF80(
                 *a1,
                 216,
                 (unsigned int)&v53,
                 **(unsigned __int16 **)(a2 + 48),
                 *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
                 v33,
                 v32);
      if ( v53 )
      {
LABEL_24:
        v50 = result;
        sub_B91220((__int64)&v53, v53);
        return v50;
      }
      return result;
    }
  }
  v17 = 1;
  do
  {
LABEL_29:
    while ( *(_DWORD *)(v15 + 8) != v7 )
    {
      v15 = *(_QWORD *)(v15 + 32);
      if ( !v15 )
        goto LABEL_31;
    }
    if ( !v17 )
      return 0;
    v34 = *(_QWORD *)(v15 + 32);
    if ( !v34 )
      goto LABEL_32;
    if ( v7 == *(_DWORD *)(v34 + 8) )
      return 0;
    v15 = *(_QWORD *)(v34 + 32);
    v17 = 0;
  }
  while ( v15 );
LABEL_31:
  if ( v17 == 1 )
    return 0;
LABEL_32:
  v35 = *(_DWORD **)(v6 + 40);
  if ( *(_DWORD *)(*(_QWORD *)v35 + 24LL) != 3 )
    return 0;
  if ( v5 != 4 )
    return 0;
  v36 = *(_QWORD *)(*(_QWORD *)v35 + 40LL);
  v46 = *(_QWORD *)v35;
  v49 = v35[2];
  if ( !sub_3280B30(
          (__int64)&v51,
          *(unsigned __int16 *)(*(_QWORD *)(v36 + 40) + 96LL),
          *(_QWORD *)(*(_QWORD *)(v36 + 40) + 104LL)) )
    return 0;
  v38 = v46;
  v53 = *(_QWORD *)(a2 + 80);
  if ( v53 )
  {
    sub_B96E90((__int64)&v53, v53, 1);
    v38 = v46;
    v36 = *(_QWORD *)(v46 + 40);
  }
  v39 = *a1;
  v54 = *(_DWORD *)(a2 + 72);
  v40 = (unsigned __int16 *)(*(_QWORD *)(v38 + 48) + 16LL * v49);
  *((_QWORD *)&v44 + 1) = v9;
  *(_QWORD *)&v44 = v8;
  *(_QWORD *)&v41 = sub_3406EB0(v39, 4, (unsigned int)&v53, *v40, *((_QWORD *)v40 + 1), v37, *(_OWORD *)v36, v44);
  result = sub_33FAF80(
             *a1,
             216,
             (unsigned int)&v53,
             **(unsigned __int16 **)(a2 + 48),
             *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
             v42,
             v41);
  if ( v53 )
    goto LABEL_24;
  return result;
}
