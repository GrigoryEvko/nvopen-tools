// Function: sub_12A38F0
// Address: 0x12a38f0
//
__int64 __fastcall sub_12A38F0(__int64 a1, __int64 a2, char *a3, __int64 a4, int a5, char a6)
{
  unsigned __int64 v10; // r13
  unsigned __int64 v11; // r15
  unsigned int v12; // esi
  __int64 v13; // rdi
  unsigned int v14; // ecx
  __int64 *v15; // rax
  __int64 v16; // rdx
  __int64 result; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  _QWORD *v20; // rax
  char v21; // r8
  unsigned int v22; // ecx
  unsigned int v23; // eax
  unsigned int v24; // eax
  int v25; // r10d
  __int64 *v26; // r9
  int v27; // ecx
  int v28; // ecx
  int v29; // eax
  int v30; // esi
  __int64 v31; // rdi
  unsigned int v32; // edx
  __int64 v33; // r8
  int v34; // r10d
  __int64 *v35; // r9
  int v36; // eax
  int v37; // edx
  __int64 v38; // rdi
  int v39; // r9d
  unsigned int v40; // r13d
  __int64 *v41; // r8
  __int64 v42; // rsi
  char v43; // [rsp+18h] [rbp-88h]
  _QWORD v46[2]; // [rsp+30h] [rbp-70h] BYREF
  __int16 v47; // [rsp+40h] [rbp-60h]
  __int64 v48[2]; // [rsp+50h] [rbp-50h] BYREF
  _QWORD v49[8]; // [rsp+60h] [rbp-40h] BYREF

  v10 = *(_QWORD *)(a2 + 120);
  if ( sub_127B420(v10) || a6 )
  {
    v11 = a4;
    goto LABEL_4;
  }
  sub_127A040(*(_QWORD *)(a1 + 32) + 8LL, v10);
  v18 = -1;
  v48[0] = (__int64)v49;
  if ( a3 )
    v18 = (__int64)&a3[strlen(a3)];
  sub_12A27A0(v48, a3, v18);
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v48[1]) <= 4 )
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490(v48, ".addr", 5, v19);
  v46[0] = "tmp";
  v47 = 259;
  v20 = sub_127FE40((_QWORD *)a1, v10, (__int64)v46);
  v11 = (unsigned __int64)v20;
  v47 = 257;
  if ( *(_BYTE *)v48[0] )
  {
    v46[0] = v48[0];
    LOBYTE(v47) = 3;
  }
  sub_164B780(v20, v46);
  if ( (*(_BYTE *)(a2 + 88) & 4) != 0 )
    goto LABEL_26;
  if ( (*(_BYTE *)(v10 + 140) & 0xFB) != 8 )
    goto LABEL_19;
  if ( (sub_8D4C10(v10, dword_4F077C4 != 2) & 2) == 0 || *(char *)(a2 + 169) >= 0 )
  {
LABEL_26:
    if ( (*(_BYTE *)(v10 + 140) & 0xFB) == 8 )
    {
      v23 = (unsigned int)sub_8D4C10(v10, dword_4F077C4 != 2) >> 1;
      v21 = v23 & 1;
      if ( *(char *)(v10 + 142) >= 0 && *(_BYTE *)(v10 + 140) == 12 )
      {
        v43 = v23 & 1;
        v24 = sub_8D4AB0(v10);
        v21 = v43;
        v22 = v24;
        goto LABEL_21;
      }
LABEL_20:
      v22 = *(_DWORD *)(v10 + 136);
LABEL_21:
      sub_1280F50((__int64 *)a1, a4, v11, v22, v21);
      goto LABEL_22;
    }
LABEL_19:
    v21 = 0;
    goto LABEL_20;
  }
LABEL_22:
  if ( (_QWORD *)v48[0] != v49 )
    j_j___libc_free_0(v48[0], v49[0] + 1LL);
LABEL_4:
  LOWORD(v49[0]) = 257;
  if ( *a3 )
  {
    v48[0] = (__int64)a3;
    LOBYTE(v49[0]) = 3;
  }
  sub_164B780(a4, v48);
  if ( sub_12A2A10(a1, a2) )
    sub_127B550("unexpected: declaration for variable already exists!", (_DWORD *)(a2 + 64), 1);
  v12 = *(_DWORD *)(a1 + 24);
  if ( !v12 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_40;
  }
  v13 = *(_QWORD *)(a1 + 8);
  v14 = (v12 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v15 = (__int64 *)(v13 + 16LL * v14);
  v16 = *v15;
  if ( a2 == *v15 )
    goto LABEL_10;
  v25 = 1;
  v26 = 0;
  while ( v16 != -8 )
  {
    if ( !v26 && v16 == -16 )
      v26 = v15;
    v14 = (v12 - 1) & (v25 + v14);
    v15 = (__int64 *)(v13 + 16LL * v14);
    v16 = *v15;
    if ( a2 == *v15 )
      goto LABEL_10;
    ++v25;
  }
  v27 = *(_DWORD *)(a1 + 16);
  if ( v26 )
    v15 = v26;
  ++*(_QWORD *)a1;
  v28 = v27 + 1;
  if ( 4 * v28 >= 3 * v12 )
  {
LABEL_40:
    sub_12A2850(a1, 2 * v12);
    v29 = *(_DWORD *)(a1 + 24);
    if ( v29 )
    {
      v30 = v29 - 1;
      v31 = *(_QWORD *)(a1 + 8);
      v32 = (v29 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v28 = *(_DWORD *)(a1 + 16) + 1;
      v15 = (__int64 *)(v31 + 16LL * v32);
      v33 = *v15;
      if ( a2 != *v15 )
      {
        v34 = 1;
        v35 = 0;
        while ( v33 != -8 )
        {
          if ( v33 == -16 && !v35 )
            v35 = v15;
          v32 = v30 & (v34 + v32);
          v15 = (__int64 *)(v31 + 16LL * v32);
          v33 = *v15;
          if ( a2 == *v15 )
            goto LABEL_36;
          ++v34;
        }
        if ( v35 )
          v15 = v35;
      }
      goto LABEL_36;
    }
    goto LABEL_69;
  }
  if ( v12 - *(_DWORD *)(a1 + 20) - v28 <= v12 >> 3 )
  {
    sub_12A2850(a1, v12);
    v36 = *(_DWORD *)(a1 + 24);
    if ( v36 )
    {
      v37 = v36 - 1;
      v38 = *(_QWORD *)(a1 + 8);
      v39 = 1;
      v40 = (v36 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v41 = 0;
      v28 = *(_DWORD *)(a1 + 16) + 1;
      v15 = (__int64 *)(v38 + 16LL * v40);
      v42 = *v15;
      if ( a2 != *v15 )
      {
        while ( v42 != -8 )
        {
          if ( v42 == -16 && !v41 )
            v41 = v15;
          v40 = v37 & (v39 + v40);
          v15 = (__int64 *)(v38 + 16LL * v40);
          v42 = *v15;
          if ( a2 == *v15 )
            goto LABEL_36;
          ++v39;
        }
        if ( v41 )
          v15 = v41;
      }
      goto LABEL_36;
    }
LABEL_69:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_36:
  *(_DWORD *)(a1 + 16) = v28;
  if ( *v15 != -8 )
    --*(_DWORD *)(a1 + 20);
  *v15 = a2;
  v15[1] = 0;
LABEL_10:
  v15[1] = v11;
  result = dword_4D046B4;
  if ( dword_4D046B4 )
    return sub_12A2460(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 384LL), a2, v11, a5, a1 + 48);
  return result;
}
