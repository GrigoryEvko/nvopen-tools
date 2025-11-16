// Function: sub_19E54E0
// Address: 0x19e54e0
//
_QWORD *__fastcall sub_19E54E0(__int64 a1, __int64 a2)
{
  __int64 *v3; // rbx
  __int64 v4; // rax
  __int64 *i; // r15
  _QWORD *result; // rax
  __int64 *v7; // r13
  __int64 v8; // rsi
  int v9; // edx
  int v10; // edx
  __int64 v11; // rdi
  unsigned int v12; // ecx
  __int64 *v13; // rax
  __int64 v14; // r10
  unsigned int v15; // ecx
  __int64 v16; // rdx
  __int64 v17; // rax
  int v18; // eax
  int v19; // r9d
  __int64 *v20; // [rsp+0h] [rbp-50h] BYREF

  v3 = *(__int64 **)(a2 + 72);
  if ( v3 == *(__int64 **)(a2 + 64) )
    v4 = *(unsigned int *)(a2 + 84);
  else
    v4 = *(unsigned int *)(a2 + 80);
  for ( i = &v3[v4]; i != v3; ++v3 )
  {
    if ( (unsigned __int64)*v3 < 0xFFFFFFFFFFFFFFFELL )
      break;
  }
  result = sub_19E54A0(&v20, (__int64 *)(a2 + 56));
  v7 = v20;
  if ( v20 != v3 )
  {
    while ( 1 )
    {
      v8 = *v3;
      if ( *(_BYTE *)(*v3 + 16) > 0x17u )
        break;
LABEL_12:
      ++v3;
      for ( result = sub_1412190(a1 + 2096, v8); i != v3; ++v3 )
      {
        if ( (unsigned __int64)*v3 < 0xFFFFFFFFFFFFFFFELL )
          break;
      }
      if ( v7 == v3 )
        return result;
    }
    v9 = *(_DWORD *)(a1 + 2416);
    if ( v9 )
    {
      v10 = v9 - 1;
      v11 = *(_QWORD *)(a1 + 2400);
      v12 = v10 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v13 = (__int64 *)(v11 + 16LL * v12);
      v14 = *v13;
      if ( v8 == *v13 )
      {
LABEL_10:
        v15 = *((_DWORD *)v13 + 2);
        v16 = 1LL << v15;
        v17 = 8LL * (v15 >> 6);
LABEL_11:
        *(_QWORD *)(*(_QWORD *)(a1 + 2336) + v17) |= v16;
        goto LABEL_12;
      }
      v18 = 1;
      while ( v14 != -8 )
      {
        v19 = v18 + 1;
        v12 = v10 & (v18 + v12);
        v13 = (__int64 *)(v11 + 16LL * v12);
        v14 = *v13;
        if ( v8 == *v13 )
          goto LABEL_10;
        v18 = v19;
      }
    }
    v16 = 1;
    v17 = 0;
    goto LABEL_11;
  }
  return result;
}
