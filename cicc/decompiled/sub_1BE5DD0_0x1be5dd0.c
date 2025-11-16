// Function: sub_1BE5DD0
// Address: 0x1be5dd0
//
__int64 *__fastcall sub_1BE5DD0(__int64 a1, __int64 a2)
{
  char v4; // r13
  __int64 v5; // r14
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rax
  char v10; // r8
  __int64 v11; // rdi
  int v12; // esi
  unsigned int v13; // edx
  __int64 *result; // rax
  __int64 v15; // r9
  __int64 v16; // rbx
  __int64 i; // r15
  unsigned int v18; // esi
  unsigned int v19; // edx
  int v20; // edi
  unsigned int v21; // r9d
  __int64 v22; // rax
  __int64 *v23; // r14
  _QWORD *v24; // rax
  _QWORD *v25; // r13
  __int64 v26; // rdi
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rsi
  __int64 v30; // rsi
  __int64 v31; // rdx
  unsigned __int8 *v32; // rsi
  __int64 *v33; // r14
  __int64 v34; // rsi
  __int64 v35; // rsi
  unsigned __int8 *v36; // rsi
  __int64 v37; // r8
  __int64 v38; // rdi
  int v39; // esi
  __int64 v40; // rdi
  int v41; // esi
  __int64 v42; // r9
  unsigned int v43; // edx
  __int64 *v44; // rax
  __int64 v45; // r10
  int v46; // r11d
  __int64 *v47; // r10
  __int64 v48; // rax
  __int64 v49; // r8
  int v50; // esi
  unsigned int v51; // edx
  __int64 v52; // rdi
  __int64 v53; // r8
  int v54; // edi
  unsigned int v55; // edx
  __int64 v56; // rsi
  int v57; // r10d
  __int64 *v58; // r9
  int v59; // esi
  int v60; // eax
  int v61; // eax
  int v62; // r10d
  int v63; // r11d
  __int64 v64; // [rsp+10h] [rbp-70h]
  __int64 *v65; // [rsp+18h] [rbp-68h]
  unsigned __int8 *v66; // [rsp+28h] [rbp-58h] BYREF
  __int64 v67[2]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v68; // [rsp+40h] [rbp-40h]

  v4 = *(_BYTE *)(a2 + 16);
  if ( v4 && !*(_DWORD *)(a2 + 8) )
    v4 = *(_DWORD *)(a2 + 12) != 0;
  v5 = *(_QWORD *)(a2 + 56);
  v6 = *(_QWORD *)(a2 + 64);
  if ( !v5 )
    goto LABEL_10;
  v7 = sub_1BE23D0(a1);
  if ( *(_DWORD *)(v7 + 64) == 1 )
  {
    v8 = **(_QWORD **)(v7 + 56);
    if ( v8 )
    {
      v9 = sub_1BE2380(v8);
      if ( v9 == v5 )
      {
        v48 = sub_1BE23A0(v9);
        if ( *(_DWORD *)(v48 + 88) == 1 )
        {
          if ( **(_QWORD **)(v48 + 80) )
            goto LABEL_10;
        }
      }
    }
  }
  if ( v4 && !*(_DWORD *)(a1 + 64) )
    goto LABEL_10;
  v6 = sub_1BE5A10(a1, a2 + 56);
  v22 = *(_QWORD *)(a2 + 176);
  *(_QWORD *)(v22 + 8) = v6;
  *(_QWORD *)(v22 + 16) = v6 + 40;
  v23 = *(__int64 **)(a2 + 176);
  v68 = 257;
  v24 = sub_1648A60(56, 0);
  v25 = v24;
  if ( v24 )
    sub_15F82A0((__int64)v24, v23[3], 0);
  v26 = v23[1];
  v64 = (__int64)(v25 + 3);
  if ( v26 )
  {
    v65 = (__int64 *)v23[2];
    sub_157E9D0(v26 + 40, (__int64)v25);
    v27 = *v65;
    v28 = v25[3] & 7LL;
    v25[4] = v65;
    v27 &= 0xFFFFFFFFFFFFFFF8LL;
    v25[3] = v27 | v28;
    *(_QWORD *)(v27 + 8) = v64;
    *v65 = v64 | *v65 & 7;
  }
  sub_164B780((__int64)v25, v67);
  v29 = *v23;
  if ( *v23 )
  {
    v66 = (unsigned __int8 *)*v23;
    sub_1623A60((__int64)&v66, v29, 2);
    v30 = v25[6];
    v31 = (__int64)(v25 + 6);
    if ( v30 )
    {
      sub_161E7C0((__int64)(v25 + 6), v30);
      v31 = (__int64)(v25 + 6);
    }
    v32 = v66;
    v25[6] = v66;
    if ( v32 )
      sub_1623210((__int64)&v66, v32, v31);
  }
  v33 = *(__int64 **)(a2 + 176);
  v33[1] = v25[5];
  v33[2] = v64;
  v34 = v25[6];
  v67[0] = v34;
  if ( v34 )
  {
    sub_1623A60((__int64)v67, v34, 2);
    v35 = *v33;
    if ( !*v33 )
      goto LABEL_38;
    goto LABEL_37;
  }
  v35 = *v33;
  if ( *v33 )
  {
LABEL_37:
    sub_161E7C0((__int64)v33, v35);
LABEL_38:
    v36 = (unsigned __int8 *)v67[0];
    *v33 = v67[0];
    if ( v36 )
    {
      sub_1623210((__int64)v67, v36, (__int64)v33);
    }
    else if ( v67[0] )
    {
      sub_161E7C0((__int64)v67, v67[0]);
    }
  }
  v37 = *(_QWORD *)(a2 + 160);
  v38 = 0;
  v39 = *(_DWORD *)(v37 + 24);
  if ( v39 )
  {
    v40 = *(_QWORD *)(a2 + 72);
    v41 = v39 - 1;
    v42 = *(_QWORD *)(v37 + 8);
    v43 = v41 & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
    v44 = (__int64 *)(v42 + 16LL * v43);
    v45 = *v44;
    if ( v40 == *v44 )
    {
LABEL_42:
      v38 = v44[1];
    }
    else
    {
      v61 = 1;
      while ( v45 != -8 )
      {
        v63 = v61 + 1;
        v43 = v41 & (v61 + v43);
        v44 = (__int64 *)(v42 + 16LL * v43);
        v45 = *v44;
        if ( v40 == *v44 )
          goto LABEL_42;
        v61 = v63;
      }
      v38 = 0;
    }
  }
  sub_1400330(v38, v6, *(_QWORD *)(a2 + 160));
  *(_QWORD *)(a2 + 64) = v6;
LABEL_10:
  v10 = *(_BYTE *)(a2 + 88) & 1;
  if ( v10 )
  {
    v11 = a2 + 96;
    v12 = 3;
  }
  else
  {
    v18 = *(_DWORD *)(a2 + 104);
    v11 = *(_QWORD *)(a2 + 96);
    if ( !v18 )
    {
      v19 = *(_DWORD *)(a2 + 88);
      ++*(_QWORD *)(a2 + 80);
      result = 0;
      v20 = (v19 >> 1) + 1;
LABEL_20:
      v21 = 3 * v18;
      goto LABEL_21;
    }
    v12 = v18 - 1;
  }
  v13 = v12 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  result = (__int64 *)(v11 + 16LL * v13);
  v15 = *result;
  if ( a1 == *result )
    goto LABEL_13;
  v46 = 1;
  v47 = 0;
  while ( v15 != -8 )
  {
    if ( !v47 && v15 == -16 )
      v47 = result;
    v13 = v12 & (v46 + v13);
    result = (__int64 *)(v11 + 16LL * v13);
    v15 = *result;
    if ( a1 == *result )
      goto LABEL_13;
    ++v46;
  }
  v19 = *(_DWORD *)(a2 + 88);
  v21 = 12;
  v18 = 4;
  if ( v47 )
    result = v47;
  ++*(_QWORD *)(a2 + 80);
  v20 = (v19 >> 1) + 1;
  if ( !v10 )
  {
    v18 = *(_DWORD *)(a2 + 104);
    goto LABEL_20;
  }
LABEL_21:
  if ( 4 * v20 >= v21 )
  {
    sub_1BE5630(a2 + 80, 2 * v18);
    if ( (*(_BYTE *)(a2 + 88) & 1) != 0 )
    {
      v49 = a2 + 96;
      v50 = 3;
    }
    else
    {
      v59 = *(_DWORD *)(a2 + 104);
      v49 = *(_QWORD *)(a2 + 96);
      if ( !v59 )
        goto LABEL_95;
      v50 = v59 - 1;
    }
    v51 = v50 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    result = (__int64 *)(v49 + 16LL * v51);
    v52 = *result;
    if ( a1 != *result )
    {
      v62 = 1;
      v58 = 0;
      while ( v52 != -8 )
      {
        if ( v52 == -16 && !v58 )
          v58 = result;
        v51 = v50 & (v62 + v51);
        result = (__int64 *)(v49 + 16LL * v51);
        v52 = *result;
        if ( a1 == *result )
          goto LABEL_60;
        ++v62;
      }
      goto LABEL_66;
    }
LABEL_60:
    v19 = *(_DWORD *)(a2 + 88);
    goto LABEL_23;
  }
  if ( v18 - *(_DWORD *)(a2 + 92) - v20 <= v18 >> 3 )
  {
    sub_1BE5630(a2 + 80, v18);
    if ( (*(_BYTE *)(a2 + 88) & 1) != 0 )
    {
      v53 = a2 + 96;
      v54 = 3;
      goto LABEL_63;
    }
    v60 = *(_DWORD *)(a2 + 104);
    v53 = *(_QWORD *)(a2 + 96);
    if ( v60 )
    {
      v54 = v60 - 1;
LABEL_63:
      v55 = v54 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      result = (__int64 *)(v53 + 16LL * v55);
      v56 = *result;
      if ( a1 != *result )
      {
        v57 = 1;
        v58 = 0;
        while ( v56 != -8 )
        {
          if ( v56 == -16 && !v58 )
            v58 = result;
          v55 = v54 & (v57 + v55);
          result = (__int64 *)(v53 + 16LL * v55);
          v56 = *result;
          if ( a1 == *result )
            goto LABEL_60;
          ++v57;
        }
LABEL_66:
        if ( v58 )
          result = v58;
        goto LABEL_60;
      }
      goto LABEL_60;
    }
LABEL_95:
    *(_DWORD *)(a2 + 88) = (2 * (*(_DWORD *)(a2 + 88) >> 1) + 2) | *(_DWORD *)(a2 + 88) & 1;
    BUG();
  }
LABEL_23:
  *(_DWORD *)(a2 + 88) = (2 * (v19 >> 1) + 2) | v19 & 1;
  if ( *result != -8 )
    --*(_DWORD *)(a2 + 92);
  *result = a1;
  result[1] = 0;
LABEL_13:
  result[1] = v6;
  *(_QWORD *)(a2 + 56) = a1;
  v16 = *(_QWORD *)(a1 + 120);
  for ( i = a1 + 112; i != v16; v16 = *(_QWORD *)(v16 + 8) )
  {
    if ( !v16 )
      BUG();
    result = (__int64 *)(*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)(v16 - 8) + 16LL))(v16 - 8, a2);
  }
  return result;
}
