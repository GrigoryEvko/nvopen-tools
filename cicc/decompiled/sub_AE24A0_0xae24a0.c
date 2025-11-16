// Function: sub_AE24A0
// Address: 0xae24a0
//
__int64 __fastcall sub_AE24A0(__int64 a1, char a2, unsigned int a3, unsigned __int8 a4, unsigned __int8 a5)
{
  __int64 v5; // r15
  __int64 v6; // r14
  __int64 v8; // r13
  __int64 v9; // rsi
  __int64 *v10; // r8
  __int64 v11; // rdi
  __int64 v12; // r9
  __int64 *v13; // rbx
  __int64 *v14; // r10
  __int64 result; // rax
  __int64 *v16; // rcx
  __int64 *v17; // rdx
  __int64 v18; // r12
  __int64 v19; // [rsp+8h] [rbp-38h]

  v5 = a5;
  v6 = a4;
  if ( a2 == 105 )
  {
    v8 = a1 + 64;
  }
  else
  {
    v8 = a1 + 176;
    if ( a2 != 118 )
    {
      v8 = a1 + 128;
      if ( a2 != 102 )
        BUG();
    }
  }
  v9 = *(unsigned int *)(v8 + 8);
  v10 = *(__int64 **)v8;
  v11 = v9;
  LODWORD(v12) = *(_DWORD *)(v8 + 8);
  v13 = *(__int64 **)v8;
  v14 = (__int64 *)(*(_QWORD *)v8 + 8 * v9);
  for ( result = (8 * v9) >> 3; result > 0; result >>= 1 )
  {
    while ( 1 )
    {
      v16 = &v13[result >> 1];
      if ( *(_DWORD *)v16 >= a3 )
        break;
      v13 = v16 + 1;
      result = result - (result >> 1) - 1;
      if ( result <= 0 )
        goto LABEL_7;
    }
  }
LABEL_7:
  if ( v14 == v13 )
  {
    result = *(unsigned int *)(v8 + 12);
    v18 = (v5 << 40) | (v6 << 32) | a3;
    if ( v9 + 1 > (unsigned __int64)result )
    {
      sub_C8D5F0(v8, v8 + 16, v9 + 1, 8);
      result = *(_QWORD *)v8;
      v13 = (__int64 *)(*(_QWORD *)v8 + 8LL * *(unsigned int *)(v8 + 8));
    }
    *v13 = v18;
    ++*(_DWORD *)(v8 + 8);
  }
  else
  {
    if ( *(_DWORD *)v13 != a3 )
    {
      result = *(unsigned int *)(v8 + 12);
      if ( v9 + 1 > (unsigned __int64)result )
      {
        v19 = *(_QWORD *)v8;
        sub_C8D5F0(v8, v8 + 16, v9 + 1, 8);
        result = *(_QWORD *)v8;
        v12 = *(unsigned int *)(v8 + 8);
        v11 = v12;
        v13 = (__int64 *)((char *)v13 + *(_QWORD *)v8 - v19);
        v14 = (__int64 *)(*(_QWORD *)v8 + 8 * v12);
        v10 = *(__int64 **)v8;
      }
      v17 = &v10[v11 - 1];
      if ( v14 )
      {
        result = *v17;
        *v14 = *v17;
        v10 = *(__int64 **)v8;
        v12 = *(unsigned int *)(v8 + 8);
        v11 = v12;
        v17 = (__int64 *)(*(_QWORD *)v8 + 8 * v12 - 8);
      }
      if ( v13 != v17 )
      {
        result = (__int64)memmove((char *)v10 + v11 * 8 - ((char *)v17 - (char *)v13), v13, (char *)v17 - (char *)v13);
        LODWORD(v12) = *(_DWORD *)(v8 + 8);
      }
      *(_DWORD *)(v8 + 8) = v12 + 1;
      *(_DWORD *)v13 = a3;
    }
    *((_BYTE *)v13 + 4) = v6;
    *((_BYTE *)v13 + 5) = v5;
  }
  return result;
}
