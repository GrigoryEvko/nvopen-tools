// Function: sub_123F610
// Address: 0x123f610
//
__int64 __fastcall sub_123F610(__int64 a1)
{
  __int64 v1; // r13
  unsigned int v3; // r12d
  int v5; // eax
  char v6; // r15
  __int64 v7; // rdx
  __int64 v8; // r8
  __int64 v9; // r15
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // r8
  __int64 *v15; // r9
  __int64 v16; // r13
  __int64 v17; // rdi
  __int64 v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // rdi
  __int64 v24; // r13
  const char *v25; // rax
  unsigned __int64 v26; // rsi
  int v27; // eax
  int v28; // esi
  __int64 v29; // rdi
  int v30; // r11d
  unsigned int v31; // eax
  __int64 *v32; // rcx
  __int64 v33; // r10
  __int64 *v34; // rax
  __int64 *v35; // r14
  __int64 v36; // rdi
  __int64 v37; // rax
  __int64 v38; // r13
  __int64 v39; // rsi
  __int64 v40; // rbx
  __int64 v41; // r14
  _QWORD *v42; // rax
  unsigned int *v43; // rdx
  __int64 *v44; // [rsp+0h] [rbp-80h]
  __int64 v45; // [rsp+8h] [rbp-78h]
  __int64 v46; // [rsp+8h] [rbp-78h]
  unsigned int v47; // [rsp+14h] [rbp-6Ch] BYREF
  __int64 *v48; // [rsp+18h] [rbp-68h] BYREF
  unsigned int *v49[4]; // [rsp+20h] [rbp-60h] BYREF
  char v50; // [rsp+40h] [rbp-40h]
  char v51; // [rsp+41h] [rbp-3Fh]

  v1 = a1 + 176;
  v47 = 0;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  if ( (unsigned __int8)sub_120BD00(a1, &v47) )
    return 1;
  v3 = sub_120AFE0(a1, 3, "expected '=' here");
  if ( (_BYTE)v3 )
    return 1;
  v5 = *(_DWORD *)(a1 + 240);
  if ( v5 == 527 )
  {
    v51 = 1;
    v25 = "unexpected type in metadata definition";
LABEL_33:
    v49[0] = (unsigned int *)v25;
    v26 = *(_QWORD *)(a1 + 232);
    v50 = 3;
    sub_11FD800(v1, v26, (__int64)v49, 1);
    return 1;
  }
  v6 = 0;
  if ( v5 == 406 )
  {
    v6 = 1;
    v5 = sub_1205200(v1);
    *(_DWORD *)(a1 + 240) = v5;
  }
  if ( v5 == 511 )
  {
    if ( (unsigned __int8)sub_122E1E0(a1, (__int64 *)&v48, v6) )
      return 1;
  }
  else if ( (unsigned __int8)sub_120AFE0(a1, 14, "Expected '!' here")
         || (unsigned __int8)sub_1225770((__int64 **)a1, &v48, v6) )
  {
    return 1;
  }
  v7 = *(_QWORD *)(a1 + 1064);
  v8 = a1 + 1056;
  v9 = a1 + 1056;
  if ( v7 )
  {
    do
    {
      while ( 1 )
      {
        v10 = *(_QWORD *)(v7 + 16);
        v11 = *(_QWORD *)(v7 + 24);
        if ( *(_DWORD *)(v7 + 32) >= v47 )
          break;
        v7 = *(_QWORD *)(v7 + 24);
        if ( !v11 )
          goto LABEL_16;
      }
      v9 = v7;
      v7 = *(_QWORD *)(v7 + 16);
    }
    while ( v10 );
LABEL_16:
    if ( v9 != v8 && v47 >= *(_DWORD *)(v9 + 32) )
    {
      v15 = v48;
      v16 = *(_QWORD *)(v9 + 40);
      if ( *(_BYTE *)v48 == 30 )
      {
        v27 = *(_DWORD *)(a1 + 920);
        v49[0] = *(unsigned int **)(v9 + 40);
        if ( v27 )
        {
          v28 = v27 - 1;
          v29 = *(_QWORD *)(a1 + 904);
          v30 = 1;
          v31 = (v27 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
          v32 = (__int64 *)(v29 + 40LL * v31);
          v33 = *v32;
          if ( v16 == *v32 )
          {
LABEL_36:
            v34 = (__int64 *)v32[1];
            v44 = &v34[*((unsigned int *)v32 + 4)];
            if ( v44 != v34 )
            {
              v35 = (__int64 *)v32[1];
              do
              {
                v36 = *v35;
                v46 = v8;
                ++v35;
                sub_B99FD0(v36, 0x26u, (__int64)v15);
                v15 = v48;
                v8 = v46;
              }
              while ( v44 != v35 );
            }
            goto LABEL_27;
          }
          while ( v33 != -4096 )
          {
            if ( v33 == -8192 && !v7 )
              v7 = (__int64)v32;
            v31 = v28 & (v30 + v31);
            v32 = (__int64 *)(v29 + 40LL * v31);
            v33 = *v32;
            if ( v16 == *v32 )
              goto LABEL_36;
            ++v30;
          }
          if ( !v7 )
            v7 = (__int64)v32;
        }
        v42 = sub_123F470(a1 + 896, v49, (_QWORD *)v7);
        v43 = v49[0];
        v8 = a1 + 1056;
        v42[2] = 0x200000000LL;
        *v42 = v43;
        v15 = v48;
        v42[1] = v42 + 3;
      }
LABEL_27:
      v17 = *(_QWORD *)(v16 + 8);
      if ( (v17 & 4) != 0 )
      {
        v45 = v8;
        sub_BA6110((const __m128i *)(v17 & 0xFFFFFFFFFFFFFFF8LL), v15);
        v8 = v45;
      }
      v18 = v8;
      v19 = sub_220F330(v9, v8);
      v23 = *(_QWORD *)(v19 + 40);
      v24 = v19;
      if ( v23 )
        sub_BA65D0(v23, v18, v20, v21, v22);
      j_j___libc_free_0(v24, 56);
      --*(_QWORD *)(a1 + 1088);
      return v3;
    }
  }
  v12 = *(_QWORD *)(a1 + 1016);
  v13 = a1 + 1008;
  if ( v12 )
  {
    v14 = a1 + 1008;
    do
    {
      if ( *(_DWORD *)(v12 + 32) < v47 )
      {
        v12 = *(_QWORD *)(v12 + 24);
      }
      else
      {
        v14 = v12;
        v12 = *(_QWORD *)(v12 + 16);
      }
    }
    while ( v12 );
    if ( v14 != v13 && v47 >= *(_DWORD *)(v14 + 32) )
    {
      v51 = 1;
      v25 = "Metadata id is already used";
      goto LABEL_33;
    }
  }
  else
  {
    v14 = a1 + 1008;
  }
  v49[0] = &v47;
  v37 = sub_121B4C0((_QWORD *)(a1 + 1000), v14, v49);
  v38 = (__int64)v48;
  v39 = *(_QWORD *)(v37 + 40);
  v40 = v37;
  v41 = v37 + 40;
  if ( v39 )
    sub_B91220(v37 + 40, v39);
  *(_QWORD *)(v40 + 40) = v38;
  if ( v38 )
    sub_B96E90(v41, v38, 1);
  return v3;
}
