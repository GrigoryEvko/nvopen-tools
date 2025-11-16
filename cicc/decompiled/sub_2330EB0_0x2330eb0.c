// Function: sub_2330EB0
// Address: 0x2330eb0
//
__int64 __fastcall sub_2330EB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v7; // rsi
  __int64 *v8; // rdx
  __int64 v9; // rcx
  __int64 **v10; // rax
  __int64 *v11; // rax
  __int64 *v13; // rax

  if ( *(_BYTE *)(a1 + 28) )
  {
    v7 = *(__int64 **)(a1 + 8);
    v8 = &v7[*(unsigned int *)(a1 + 20)];
    v9 = *(unsigned int *)(a1 + 20);
    v10 = (__int64 **)v7;
    if ( v7 != v8 )
    {
      while ( (__int64 *)a2 != *v10 )
      {
        if ( v8 == (__int64 *)++v10 )
          goto LABEL_7;
      }
      v9 = (unsigned int)(v9 - 1);
      *(_DWORD *)(a1 + 20) = v9;
      v8 = (__int64 *)v7[v9];
      *v10 = v8;
      ++*(_QWORD *)a1;
    }
  }
  else
  {
    v13 = sub_C8CA60(a1, a2);
    if ( v13 )
    {
      *v13 = -2;
      ++*(_DWORD *)(a1 + 24);
      ++*(_QWORD *)a1;
    }
  }
LABEL_7:
  if ( !*(_BYTE *)(a1 + 76) )
    goto LABEL_14;
  v11 = *(__int64 **)(a1 + 56);
  v9 = *(unsigned int *)(a1 + 68);
  v8 = &v11[v9];
  if ( v11 == v8 )
  {
LABEL_13:
    if ( (unsigned int)v9 >= *(_DWORD *)(a1 + 64) )
    {
LABEL_14:
      sub_C8CC70(a1 + 48, a2, (__int64)v8, v9, a5, a6);
      return a1;
    }
    *(_DWORD *)(a1 + 68) = v9 + 1;
    *v8 = a2;
    ++*(_QWORD *)(a1 + 48);
    return a1;
  }
  else
  {
    while ( a2 != *v11 )
    {
      if ( v8 == ++v11 )
        goto LABEL_13;
    }
    return a1;
  }
}
