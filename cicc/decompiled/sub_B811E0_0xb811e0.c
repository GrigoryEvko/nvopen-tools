// Function: sub_B811E0
// Address: 0xb811e0
//
__int64 __fastcall sub_B811E0(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  int v5; // ecx
  unsigned int v6; // edx
  __int64 *v7; // rax
  __int64 v8; // rdi
  __int64 result; // rax
  int v10; // eax
  __int64 *v11; // rbx
  __int64 *v12; // r13
  int v13; // ecx
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
    v13 = *(_DWORD *)(a1 + 424);
    v4 = *(_QWORD *)(a1 + 416);
    if ( !v13 )
      goto LABEL_8;
    v5 = v13 - 1;
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
    v10 = 1;
    while ( v8 != -4096 )
    {
      v16 = v10 + 1;
      v6 = v5 & (v10 + v6);
      v7 = (__int64 *)(v4 + 16LL * v6);
      v8 = *v7;
      if ( a2 == *v7 )
        goto LABEL_4;
      v10 = v16;
    }
  }
LABEL_8:
  v11 = *(__int64 **)(a1 + 32);
  v12 = &v11[*(unsigned int *)(a1 + 40)];
  if ( v11 == v12 )
  {
LABEL_15:
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
        result = sub_B81110(*v14, a2, 0);
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
      result = sub_B81110(*v11, a2, 0);
      if ( result )
        break;
      if ( v12 == ++v11 )
        goto LABEL_15;
    }
  }
  return result;
}
