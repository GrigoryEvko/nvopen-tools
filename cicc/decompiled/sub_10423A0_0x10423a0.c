// Function: sub_10423A0
// Address: 0x10423a0
//
__int64 *__fastcall sub_10423A0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v7; // rax
  __int64 v8; // rsi
  unsigned int v9; // ecx
  __int64 *v10; // rdx
  __int64 v11; // r8
  __int64 *v12; // r15
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 *v15; // rsi
  __int64 *v16; // rdx
  __int64 *result; // rax
  __int64 v18; // rcx
  __int64 *v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rdx
  __int64 v22; // rcx
  int v23; // edx
  int v24; // r9d

  v7 = *(unsigned int *)(a1 + 88);
  v8 = *(_QWORD *)(a1 + 72);
  if ( (_DWORD)v7 )
  {
    v9 = (v7 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v10 = (__int64 *)(v8 + 16LL * v9);
    v11 = *v10;
    if ( a3 == *v10 )
    {
LABEL_3:
      if ( v10 != (__int64 *)(v8 + 16 * v7) )
      {
        v12 = (__int64 *)v10[1];
        goto LABEL_5;
      }
    }
    else
    {
      v23 = 1;
      while ( v11 != -4096 )
      {
        v24 = v23 + 1;
        v9 = (v7 - 1) & (v23 + v9);
        v10 = (__int64 *)(v8 + 16LL * v9);
        v11 = *v10;
        if ( a3 == *v10 )
          goto LABEL_3;
        v23 = v24;
      }
    }
  }
  v12 = 0;
LABEL_5:
  v13 = *a4;
  v14 = *(_QWORD *)(a2 + 32);
  *(_QWORD *)(a2 + 40) = a4;
  v13 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)(a2 + 32) = v13 | v14 & 7;
  *(_QWORD *)(v13 + 8) = a2 + 32;
  *a4 = *a4 & 7 | (a2 + 32);
  if ( *(_BYTE *)a2 == 26 )
    goto LABEL_6;
  v19 = (__int64 *)sub_1041AC0(a1, a3);
  if ( a4 != v12 )
  {
    if ( *((_BYTE *)a4 - 32) == 27 )
      goto LABEL_23;
    while ( *((_BYTE *)a4 - 32) != 27 )
    {
      a4 = (__int64 *)a4[1];
      if ( v12 == a4 )
        goto LABEL_19;
      if ( !a4 )
        BUG();
    }
    if ( v12 != a4 )
    {
LABEL_23:
      v22 = a4[2];
      *(_QWORD *)(a2 + 56) = a4 + 2;
      v22 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(a2 + 48) = v22 | *(_QWORD *)(a2 + 48) & 7LL;
      *(_QWORD *)(v22 + 8) = a2 + 48;
      a4[2] = a4[2] & 7 | (a2 + 48);
LABEL_6:
      if ( *(_BYTE *)(a1 + 164) )
        goto LABEL_7;
LABEL_20:
      result = sub_C8CA60(a1 + 136, a3);
      if ( result )
      {
        *result = -2;
        ++*(_DWORD *)(a1 + 160);
        ++*(_QWORD *)(a1 + 136);
      }
      return result;
    }
  }
LABEL_19:
  v20 = *v19;
  v21 = *(_QWORD *)(a2 + 48);
  *(_QWORD *)(a2 + 56) = v19;
  v20 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)(a2 + 48) = v20 | v21 & 7;
  *(_QWORD *)(v20 + 8) = a2 + 48;
  *v19 = *v19 & 7 | (a2 + 48);
  if ( !*(_BYTE *)(a1 + 164) )
    goto LABEL_20;
LABEL_7:
  v15 = *(__int64 **)(a1 + 144);
  v16 = &v15[*(unsigned int *)(a1 + 156)];
  result = v15;
  if ( v15 != v16 )
  {
    while ( a3 != *result )
    {
      if ( v16 == ++result )
        return result;
    }
    v18 = (unsigned int)(*(_DWORD *)(a1 + 156) - 1);
    *(_DWORD *)(a1 + 156) = v18;
    *result = v15[v18];
    ++*(_QWORD *)(a1 + 136);
  }
  return result;
}
