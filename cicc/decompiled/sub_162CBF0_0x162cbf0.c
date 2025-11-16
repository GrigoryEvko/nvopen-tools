// Function: sub_162CBF0
// Address: 0x162cbf0
//
__int64 __fastcall sub_162CBF0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  int v4; // r12d
  __int64 v5; // r13
  unsigned int v6; // r9d
  int v7; // esi
  __int64 *v8; // rsi
  __int64 v9; // rdi
  int v10; // eax
  __int64 result; // rax
  int v12; // r12d
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rdi
  __int64 v17; // r13
  __int64 v18; // r12
  int v19; // r12d
  __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 *v22; // r8
  __int64 v23; // rcx
  int v24; // r9d
  __int64 *v25; // r10
  int v26; // eax
  int i; // [rsp+50h] [rbp-A0h]
  __int64 v28; // [rsp+68h] [rbp-88h] BYREF
  __int64 *v29; // [rsp+70h] [rbp-80h] BYREF
  __int64 v30; // [rsp+78h] [rbp-78h] BYREF
  __int64 v31; // [rsp+80h] [rbp-70h] BYREF
  __int64 v32; // [rsp+88h] [rbp-68h] BYREF
  int v33; // [rsp+90h] [rbp-60h] BYREF
  __int64 v34; // [rsp+98h] [rbp-58h] BYREF
  __int8 v35; // [rsp+A0h] [rbp-50h] BYREF
  __int8 v36[7]; // [rsp+A1h] [rbp-4Fh] BYREF
  __int64 v37; // [rsp+A8h] [rbp-48h] BYREF
  int v38; // [rsp+B0h] [rbp-40h]

  v3 = *(unsigned int *)(a1 + 8);
  v4 = *(_DWORD *)(a2 + 24);
  v28 = a1;
  v5 = *(_QWORD *)(a2 + 8);
  v29 = *(__int64 **)(a1 - 8 * v3);
  v30 = *(_QWORD *)(a1 + 8 * (1 - v3));
  v31 = *(_QWORD *)(a1 + 8 * (5 - v3));
  v32 = *(_QWORD *)(a1 + 8 * (2 - v3));
  v33 = *(_DWORD *)(a1 + 24);
  v34 = *(_QWORD *)(a1 + 8 * (3 - v3));
  v35 = *(_BYTE *)(a1 + 32);
  v36[0] = *(_BYTE *)(a1 + 33);
  v37 = *(_QWORD *)(a1 + 8 * (6 - v3));
  v38 = *(_DWORD *)(a1 + 28);
  if ( !v4 )
    goto LABEL_2;
  v12 = v4 - 1;
  v13 = v12 & sub_15B44C0((__int64 *)&v29, &v30, &v31, &v32, &v33, &v34, &v35, v36, &v37);
  v14 = (__int64 *)(v5 + 8LL * v13);
  v15 = *v14;
  if ( *v14 == -8 )
  {
LABEL_15:
    v17 = *(_QWORD *)(a2 + 8);
    LODWORD(v18) = *(_DWORD *)(a2 + 24);
    goto LABEL_16;
  }
  for ( i = 1; ; ++i )
  {
    if ( v15 != -16 )
    {
      v16 = *(unsigned int *)(v15 + 8);
      if ( v29 == *(__int64 **)(v15 - 8 * v16)
        && v30 == *(_QWORD *)(v15 + 8 * (1 - v16))
        && v31 == *(_QWORD *)(v15 + 8 * (5 - v16))
        && v32 == *(_QWORD *)(v15 + 8 * (2 - v16))
        && v33 == *(_DWORD *)(v15 + 24)
        && v34 == *(_QWORD *)(v15 + 8 * (3 - v16))
        && v35 == *(_BYTE *)(v15 + 32)
        && v36[0] == *(_BYTE *)(v15 + 33)
        && v37 == *(_QWORD *)(v15 + 8 * (6 - v16))
        && v38 == *(_DWORD *)(v15 + 28) )
      {
        break;
      }
    }
    v13 = v12 & (i + v13);
    v14 = (__int64 *)(v5 + 8LL * v13);
    v15 = *v14;
    if ( *v14 == -8 )
      goto LABEL_15;
  }
  v17 = *(_QWORD *)(a2 + 8);
  v18 = *(unsigned int *)(a2 + 24);
  if ( v14 == (__int64 *)(v17 + 8 * v18) || (result = *v14) == 0 )
  {
LABEL_16:
    if ( (_DWORD)v18 )
    {
      v19 = v18 - 1;
      v20 = *(unsigned int *)(v28 + 8);
      v29 = *(__int64 **)(v28 - 8 * v20);
      v30 = *(_QWORD *)(v28 + 8 * (1 - v20));
      v31 = *(_QWORD *)(v28 + 8 * (5 - v20));
      v32 = *(_QWORD *)(v28 + 8 * (2 - v20));
      v33 = *(_DWORD *)(v28 + 24);
      v34 = *(_QWORD *)(v28 + 8 * (3 - v20));
      v35 = *(_BYTE *)(v28 + 32);
      v36[0] = *(_BYTE *)(v28 + 33);
      v37 = *(_QWORD *)(v28 + 8 * (6 - v20));
      v38 = *(_DWORD *)(v28 + 28);
      v9 = v28;
      LODWORD(v21) = v19 & sub_15B44C0((__int64 *)&v29, &v30, &v31, &v32, &v33, &v34, &v35, v36, &v37);
      v22 = (__int64 *)(v17 + 8LL * (unsigned int)v21);
      result = v28;
      v23 = *v22;
      if ( *v22 == v28 )
        return result;
      v24 = 1;
      v8 = 0;
      while ( v23 != -8 )
      {
        if ( v23 != -16 || v8 )
          v22 = v8;
        v21 = v19 & (unsigned int)(v21 + v24);
        v25 = (__int64 *)(v17 + 8 * v21);
        v23 = *v25;
        if ( *v25 == v28 )
          return result;
        ++v24;
        v8 = v22;
        v22 = (__int64 *)(v17 + 8 * v21);
      }
      v26 = *(_DWORD *)(a2 + 16);
      v6 = *(_DWORD *)(a2 + 24);
      if ( !v8 )
        v8 = v22;
      ++*(_QWORD *)a2;
      v10 = v26 + 1;
      if ( 4 * v10 < 3 * v6 )
      {
        if ( v6 - (v10 + *(_DWORD *)(a2 + 20)) > v6 >> 3 )
          goto LABEL_5;
        v7 = v6;
LABEL_4:
        sub_15C2C50(a2, v7);
        sub_15B8E40(a2, &v28, &v29);
        v8 = v29;
        v9 = v28;
        v10 = *(_DWORD *)(a2 + 16) + 1;
LABEL_5:
        *(_DWORD *)(a2 + 16) = v10;
        if ( *v8 != -8 )
          --*(_DWORD *)(a2 + 20);
        *v8 = v9;
        return v28;
      }
LABEL_3:
      v7 = 2 * v6;
      goto LABEL_4;
    }
LABEL_2:
    ++*(_QWORD *)a2;
    v6 = 0;
    goto LABEL_3;
  }
  return result;
}
