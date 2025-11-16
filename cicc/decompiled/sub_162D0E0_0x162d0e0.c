// Function: sub_162D0E0
// Address: 0x162d0e0
//
__int64 __fastcall sub_162D0E0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  int v4; // r12d
  __int64 v5; // r13
  unsigned int v6; // r8d
  int v7; // esi
  __int64 *v8; // rcx
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
  __int64 *v22; // rsi
  int v23; // r8d
  __int64 *v24; // r9
  int v25; // eax
  int i; // [rsp+1Ch] [rbp-A4h]
  __int64 v27; // [rsp+48h] [rbp-78h] BYREF
  __int64 *v28; // [rsp+50h] [rbp-70h] BYREF
  __int64 v29; // [rsp+58h] [rbp-68h] BYREF
  __int64 v30; // [rsp+60h] [rbp-60h] BYREF
  int v31; // [rsp+68h] [rbp-58h] BYREF
  __int64 v32; // [rsp+70h] [rbp-50h] BYREF
  int v33; // [rsp+78h] [rbp-48h] BYREF
  int v34; // [rsp+7Ch] [rbp-44h] BYREF
  int v35; // [rsp+80h] [rbp-40h]

  v3 = *(unsigned int *)(a1 + 8);
  v27 = a1;
  v4 = *(_DWORD *)(a2 + 24);
  v5 = *(_QWORD *)(a2 + 8);
  v28 = *(__int64 **)(a1 - 8 * v3);
  v29 = *(_QWORD *)(a1 + 8 * (1 - v3));
  v30 = *(_QWORD *)(a1 + 8 * (2 - v3));
  v31 = *(_DWORD *)(a1 + 24);
  v32 = *(_QWORD *)(a1 + 8 * (3 - v3));
  v33 = *(unsigned __int16 *)(a1 + 32);
  v34 = *(_DWORD *)(a1 + 36);
  v35 = *(_DWORD *)(a1 + 28);
  if ( !v4 )
    goto LABEL_2;
  v12 = v4 - 1;
  v13 = v12 & sub_15B41C0((__int64 *)&v28, &v29, &v30, &v31, &v32, &v33, &v34);
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
      if ( v28 == *(__int64 **)(v15 - 8 * v16)
        && v29 == *(_QWORD *)(v15 + 8 * (1 - v16))
        && v30 == *(_QWORD *)(v15 + 8 * (2 - v16))
        && v31 == *(_DWORD *)(v15 + 24)
        && v32 == *(_QWORD *)(v15 + 8 * (3 - v16))
        && v33 == *(unsigned __int16 *)(v15 + 32)
        && v34 == *(_DWORD *)(v15 + 36)
        && v35 == *(_DWORD *)(v15 + 28) )
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
      v20 = *(unsigned int *)(v27 + 8);
      v28 = *(__int64 **)(v27 - 8 * v20);
      v29 = *(_QWORD *)(v27 + 8 * (1 - v20));
      v30 = *(_QWORD *)(v27 + 8 * (2 - v20));
      v31 = *(_DWORD *)(v27 + 24);
      v32 = *(_QWORD *)(v27 + 8 * (3 - v20));
      v33 = *(unsigned __int16 *)(v27 + 32);
      v34 = *(_DWORD *)(v27 + 36);
      v35 = *(_DWORD *)(v27 + 28);
      v9 = v27;
      LODWORD(v21) = v19 & sub_15B41C0((__int64 *)&v28, &v29, &v30, &v31, &v32, &v33, &v34);
      v22 = (__int64 *)(v17 + 8LL * (unsigned int)v21);
      result = *v22;
      if ( v27 == *v22 )
        return result;
      v23 = 1;
      v8 = 0;
      while ( result != -8 )
      {
        if ( result != -16 || v8 )
          v22 = v8;
        v21 = v19 & (unsigned int)(v21 + v23);
        v24 = (__int64 *)(v17 + 8 * v21);
        result = *v24;
        if ( *v24 == v27 )
          return result;
        ++v23;
        v8 = v22;
        v22 = (__int64 *)(v17 + 8 * v21);
      }
      v25 = *(_DWORD *)(a2 + 16);
      v6 = *(_DWORD *)(a2 + 24);
      if ( !v8 )
        v8 = v22;
      ++*(_QWORD *)a2;
      v10 = v25 + 1;
      if ( 4 * v10 < 3 * v6 )
      {
        if ( v6 - (v10 + *(_DWORD *)(a2 + 20)) > v6 >> 3 )
          goto LABEL_5;
        v7 = v6;
LABEL_4:
        sub_15C3360(a2, v7);
        sub_15B8FB0(a2, &v27, &v28);
        v8 = v28;
        v9 = v27;
        v10 = *(_DWORD *)(a2 + 16) + 1;
LABEL_5:
        *(_DWORD *)(a2 + 16) = v10;
        if ( *v8 != -8 )
          --*(_DWORD *)(a2 + 20);
        *v8 = v9;
        return v27;
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
