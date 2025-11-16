// Function: sub_138E440
// Address: 0x138e440
//
__int64 *__fastcall sub_138E440(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v5; // rsi
  unsigned int v6; // ecx
  __int64 *v7; // rax
  __int64 v8; // rdi
  int v10; // eax
  __int64 v11; // rdx
  __int64 v12; // rsi
  unsigned int v13; // ecx
  __int64 v14; // rdi
  int v15; // eax
  int v16; // r9d
  int v17; // r9d

  v3 = *(unsigned int *)(a1 + 40);
  if ( !(_DWORD)v3 )
    goto LABEL_7;
  v5 = *(_QWORD *)(a1 + 24);
  v6 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v7 = (__int64 *)(v5 + 432LL * v6);
  v8 = *v7;
  if ( a2 != *v7 )
  {
    v10 = 1;
    while ( v8 != -8 )
    {
      v17 = v10 + 1;
      v6 = (v3 - 1) & (v10 + v6);
      v7 = (__int64 *)(v5 + 432LL * v6);
      v8 = *v7;
      if ( a2 == *v7 )
        goto LABEL_3;
      v10 = v17;
    }
LABEL_7:
    sub_138D9C0(a1, a2);
    v11 = *(unsigned int *)(a1 + 40);
    v12 = *(_QWORD *)(a1 + 24);
    if ( (_DWORD)v11 )
    {
      v13 = (v11 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v7 = (__int64 *)(v12 + 432LL * v13);
      v14 = *v7;
      if ( a2 == *v7 )
        return v7 + 1;
      v15 = 1;
      while ( v14 != -8 )
      {
        v16 = v15 + 1;
        v13 = (v11 - 1) & (v15 + v13);
        v7 = (__int64 *)(v12 + 432LL * v13);
        v14 = *v7;
        if ( a2 == *v7 )
          return v7 + 1;
        v15 = v16;
      }
    }
    v7 = (__int64 *)(v12 + 432 * v11);
    return v7 + 1;
  }
LABEL_3:
  if ( v7 == (__int64 *)(v5 + 432 * v3) )
    goto LABEL_7;
  return v7 + 1;
}
