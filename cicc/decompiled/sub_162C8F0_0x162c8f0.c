// Function: sub_162C8F0
// Address: 0x162c8f0
//
__int64 __fastcall sub_162C8F0(__int64 a1, __int64 a2)
{
  int v3; // r9d
  __int64 v4; // r13
  __int64 v5; // rdx
  unsigned int v6; // r8d
  int v7; // esi
  __int64 *v8; // rcx
  __int64 v9; // rdi
  int v10; // eax
  __int64 result; // rax
  unsigned int v12; // esi
  __int64 *v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r13
  __int64 v16; // r9
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 *v19; // rsi
  int v20; // r8d
  __int64 *v21; // r10
  __int64 v22; // rax
  int v23; // eax
  int i; // [rsp+8h] [rbp-68h]
  int v25; // [rsp+10h] [rbp-60h]
  int v26; // [rsp+10h] [rbp-60h]
  __int64 v27; // [rsp+18h] [rbp-58h] BYREF
  __int64 *v28; // [rsp+20h] [rbp-50h] BYREF
  __int64 v29; // [rsp+28h] [rbp-48h] BYREF
  __int64 v30; // [rsp+30h] [rbp-40h] BYREF
  __int64 v31[7]; // [rsp+38h] [rbp-38h] BYREF

  v3 = *(_DWORD *)(a2 + 24);
  v4 = *(_QWORD *)(a2 + 8);
  v27 = a1;
  v25 = v3;
  LODWORD(v28) = *(unsigned __int16 *)(a1 + 2);
  v5 = *(unsigned int *)(a1 + 8);
  v29 = *(_QWORD *)(a1 - 8 * v5);
  v30 = *(_QWORD *)(a1 + 8 * (1 - v5));
  v31[0] = *(_QWORD *)(a1 + 8 * (2 - v5));
  if ( !v3 )
    goto LABEL_2;
  v12 = (v3 - 1) & sub_15B49E0((__int32 *)&v28, &v29, &v30, v31);
  v13 = (__int64 *)(v4 + 8LL * v12);
  v14 = *v13;
  if ( *v13 == -8 )
  {
LABEL_15:
    v15 = *(_QWORD *)(a2 + 8);
    LODWORD(v16) = *(_DWORD *)(a2 + 24);
    goto LABEL_16;
  }
  for ( i = 1; ; ++i )
  {
    if ( v14 != -16 && (_DWORD)v28 == *(unsigned __int16 *)(v14 + 2) )
    {
      v22 = *(unsigned int *)(v14 + 8);
      if ( v29 == *(_QWORD *)(v14 - 8 * v22)
        && v30 == *(_QWORD *)(v14 + 8 * (1 - v22))
        && v31[0] == *(_QWORD *)(v14 + 8 * (2 - v22)) )
      {
        break;
      }
    }
    v12 = (v25 - 1) & (i + v12);
    v13 = (__int64 *)(v4 + 8LL * v12);
    v14 = *v13;
    if ( *v13 == -8 )
      goto LABEL_15;
  }
  v15 = *(_QWORD *)(a2 + 8);
  v16 = *(unsigned int *)(a2 + 24);
  if ( v13 == (__int64 *)(v15 + 8 * v16) || (result = *v13) == 0 )
  {
LABEL_16:
    v26 = v16;
    if ( (_DWORD)v16 )
    {
      LODWORD(v28) = *(unsigned __int16 *)(v27 + 2);
      v17 = *(unsigned int *)(v27 + 8);
      v29 = *(_QWORD *)(v27 - 8 * v17);
      v30 = *(_QWORD *)(v27 + 8 * (1 - v17));
      v31[0] = *(_QWORD *)(v27 + 8 * (2 - v17));
      v9 = v27;
      LODWORD(v18) = (v16 - 1) & sub_15B49E0((__int32 *)&v28, &v29, &v30, v31);
      v19 = (__int64 *)(v15 + 8LL * (unsigned int)v18);
      result = *v19;
      if ( v27 == *v19 )
        return result;
      v20 = 1;
      v8 = 0;
      while ( result != -8 )
      {
        if ( result != -16 || v8 )
          v19 = v8;
        v18 = (v26 - 1) & (unsigned int)(v18 + v20);
        v21 = (__int64 *)(v15 + 8 * v18);
        result = *v21;
        if ( *v21 == v27 )
          return result;
        ++v20;
        v8 = v19;
        v19 = (__int64 *)(v15 + 8 * v18);
      }
      v23 = *(_DWORD *)(a2 + 16);
      v6 = *(_DWORD *)(a2 + 24);
      if ( !v8 )
        v8 = v19;
      ++*(_QWORD *)a2;
      v10 = v23 + 1;
      if ( 4 * v10 < 3 * v6 )
      {
        if ( v6 - (v10 + *(_DWORD *)(a2 + 20)) > v6 >> 3 )
          goto LABEL_5;
        v7 = v6;
LABEL_4:
        sub_15C2670(a2, v7);
        sub_15B8D30(a2, &v27, &v28);
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
