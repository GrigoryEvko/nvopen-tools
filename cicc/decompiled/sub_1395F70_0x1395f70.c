// Function: sub_1395F70
// Address: 0x1395f70
//
__int64 *__fastcall sub_1395F70(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v5; // rsi
  unsigned int v6; // ecx
  __int64 *v7; // rax
  __int64 v8; // rdi
  int v10; // eax
  unsigned int v11; // edx
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
  v7 = (__int64 *)(v5 + 424LL * v6);
  v8 = *v7;
  if ( a2 != *v7 )
  {
    v10 = 1;
    while ( v8 != -8 )
    {
      v17 = v10 + 1;
      v6 = (v3 - 1) & (v10 + v6);
      v7 = (__int64 *)(v5 + 424LL * v6);
      v8 = *v7;
      if ( a2 == *v7 )
        goto LABEL_3;
      v10 = v17;
    }
LABEL_7:
    sub_1395630(a1, a2);
    v11 = *(_DWORD *)(a1 + 40);
    v12 = *(_QWORD *)(a1 + 24);
    if ( v11 )
    {
      v13 = (v11 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v7 = (__int64 *)(v12 + 424LL * v13);
      v14 = *v7;
      if ( a2 == *v7 )
        return v7 + 1;
      v15 = 1;
      while ( v14 != -8 )
      {
        v16 = v15 + 1;
        v13 = (v11 - 1) & (v15 + v13);
        v7 = (__int64 *)(v12 + 424LL * v13);
        v14 = *v7;
        if ( a2 == *v7 )
          return v7 + 1;
        v15 = v16;
      }
    }
    v7 = (__int64 *)(v12 + 424LL * v11);
    return v7 + 1;
  }
LABEL_3:
  if ( v7 == (__int64 *)(v5 + 424 * v3) )
    goto LABEL_7;
  return v7 + 1;
}
