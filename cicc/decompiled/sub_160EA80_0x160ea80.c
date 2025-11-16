// Function: sub_160EA80
// Address: 0x160ea80
//
__int64 __fastcall sub_160EA80(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  int v5; // ecx
  unsigned int v6; // edx
  __int64 *v7; // rax
  __int64 v8; // rdi
  __int64 result; // rax
  int v10; // eax
  int v11; // eax
  __int64 *v12; // rbx
  __int64 *v13; // r13
  __int64 *v14; // rbx
  __int64 *v15; // r12
  int v16; // r8d

  if ( (*(_BYTE *)(a1 + 408) & 1) != 0 )
  {
    v4 = a1 + 416;
    v5 = 7;
  }
  else
  {
    v10 = *(_DWORD *)(a1 + 424);
    v4 = *(_QWORD *)(a1 + 416);
    if ( !v10 )
      goto LABEL_10;
    v5 = v10 - 1;
  }
  v6 = v5 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v7 = (__int64 *)(v4 + 16LL * v6);
  v8 = *v7;
  if ( a2 == *v7 )
  {
LABEL_4:
    result = v7[1];
    if ( result )
      return result;
  }
  else
  {
    v11 = 1;
    while ( v8 != -4 )
    {
      v16 = v11 + 1;
      v6 = v5 & (v11 + v6);
      v7 = (__int64 *)(v4 + 16LL * v6);
      v8 = *v7;
      if ( a2 == *v7 )
        goto LABEL_4;
      v11 = v16;
    }
  }
LABEL_10:
  v12 = *(__int64 **)(a1 + 32);
  v13 = &v12[*(unsigned int *)(a1 + 40)];
  if ( v12 == v13 )
  {
LABEL_13:
    v14 = *(__int64 **)(a1 + 112);
    v15 = &v14[*(unsigned int *)(a1 + 120)];
    if ( v14 == v15 )
    {
      return 0;
    }
    else
    {
      while ( 1 )
      {
        result = sub_160E9B0(*v14, a2, 0);
        if ( result )
          break;
        if ( v15 == ++v14 )
          return 0;
      }
    }
  }
  else
  {
    while ( 1 )
    {
      result = sub_160E9B0(*v12, a2, 0);
      if ( result )
        break;
      if ( v13 == ++v12 )
        goto LABEL_13;
    }
  }
  return result;
}
