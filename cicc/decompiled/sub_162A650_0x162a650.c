// Function: sub_162A650
// Address: 0x162a650
//
__int64 __fastcall sub_162A650(__int64 a1, __int64 a2)
{
  int v3; // r13d
  __int64 v4; // r12
  unsigned int v5; // r8d
  int v6; // esi
  __int64 *v7; // rcx
  __int64 v8; // rdi
  int v9; // eax
  __int64 result; // rax
  unsigned int v11; // r9d
  __int64 *v12; // rsi
  __int64 v13; // rdi
  __int64 v14; // r13
  __int64 v15; // r12
  int v16; // r12d
  __int64 v17; // rdx
  __int64 *v18; // rsi
  int v19; // r8d
  __int64 *v20; // r9
  int v21; // eax
  int v22; // [rsp+10h] [rbp-70h]
  __int64 v23; // [rsp+28h] [rbp-58h] BYREF
  __int64 *v24; // [rsp+30h] [rbp-50h] BYREF
  __int64 v25; // [rsp+38h] [rbp-48h] BYREF
  __int64 v26; // [rsp+40h] [rbp-40h] BYREF
  int v27[14]; // [rsp+48h] [rbp-38h] BYREF

  v3 = *(_DWORD *)(a2 + 24);
  v4 = *(_QWORD *)(a2 + 8);
  v23 = a1;
  LODWORD(v24) = *(unsigned __int16 *)(a1 + 2);
  v25 = *(_QWORD *)(a1 + 8 * (2LL - *(unsigned int *)(a1 + 8)));
  v26 = *(_QWORD *)(a1 + 32);
  *(_QWORD *)v27 = *(_QWORD *)(a1 + 48);
  if ( !v3 )
    goto LABEL_2;
  v11 = (v3 - 1) & sub_15B4F20((int *)&v24, &v25, &v26, v27, &v27[1]);
  v12 = (__int64 *)(v4 + 8LL * v11);
  v13 = *v12;
  if ( *v12 == -8 )
  {
LABEL_15:
    v14 = *(_QWORD *)(a2 + 8);
    LODWORD(v15) = *(_DWORD *)(a2 + 24);
    goto LABEL_16;
  }
  v22 = 1;
  while ( v13 == -16
       || (_DWORD)v24 != *(unsigned __int16 *)(v13 + 2)
       || v25 != *(_QWORD *)(v13 + 8 * (2LL - *(unsigned int *)(v13 + 8)))
       || v26 != *(_QWORD *)(v13 + 32)
       || *(_QWORD *)v27 != *(_QWORD *)(v13 + 48) )
  {
    v11 = (v3 - 1) & (v22 + v11);
    v12 = (__int64 *)(v4 + 8LL * v11);
    v13 = *v12;
    if ( *v12 == -8 )
      goto LABEL_15;
    ++v22;
  }
  v14 = *(_QWORD *)(a2 + 8);
  v15 = *(unsigned int *)(a2 + 24);
  if ( v12 == (__int64 *)(v14 + 8 * v15) || (result = *v12) == 0 )
  {
LABEL_16:
    if ( (_DWORD)v15 )
    {
      v16 = v15 - 1;
      LODWORD(v24) = *(unsigned __int16 *)(v23 + 2);
      v25 = *(_QWORD *)(v23 + 8 * (2LL - *(unsigned int *)(v23 + 8)));
      v26 = *(_QWORD *)(v23 + 32);
      *(_QWORD *)v27 = *(_QWORD *)(v23 + 48);
      v8 = v23;
      LODWORD(v17) = v16 & sub_15B4F20((int *)&v24, &v25, &v26, v27, &v27[1]);
      v18 = (__int64 *)(v14 + 8LL * (unsigned int)v17);
      result = *v18;
      if ( v23 == *v18 )
        return result;
      v19 = 1;
      v7 = 0;
      while ( result != -8 )
      {
        if ( result != -16 || v7 )
          v18 = v7;
        v17 = v16 & (unsigned int)(v17 + v19);
        v20 = (__int64 *)(v14 + 8 * v17);
        result = *v20;
        if ( *v20 == v23 )
          return result;
        ++v19;
        v7 = v18;
        v18 = (__int64 *)(v14 + 8 * v17);
      }
      v21 = *(_DWORD *)(a2 + 16);
      v5 = *(_DWORD *)(a2 + 24);
      if ( !v7 )
        v7 = v18;
      ++*(_QWORD *)a2;
      v9 = v21 + 1;
      if ( 4 * v9 < 3 * v5 )
      {
        if ( v5 - (v9 + *(_DWORD *)(a2 + 20)) > v5 >> 3 )
          goto LABEL_5;
        v6 = v5;
LABEL_4:
        sub_15BC460(a2, v6);
        sub_15B78B0(a2, &v23, &v24);
        v7 = v24;
        v8 = v23;
        v9 = *(_DWORD *)(a2 + 16) + 1;
LABEL_5:
        *(_DWORD *)(a2 + 16) = v9;
        if ( *v7 != -8 )
          --*(_DWORD *)(a2 + 20);
        *v7 = v8;
        return v23;
      }
LABEL_3:
      v6 = 2 * v5;
      goto LABEL_4;
    }
LABEL_2:
    ++*(_QWORD *)a2;
    v5 = 0;
    goto LABEL_3;
  }
  return result;
}
