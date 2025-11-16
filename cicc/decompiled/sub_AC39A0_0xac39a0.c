// Function: sub_AC39A0
// Address: 0xac39a0
//
__int64 __fastcall sub_AC39A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  __int64 v5; // r12
  __int64 result; // rax
  __int64 v7; // rsi
  int v8; // edx
  __int64 *v9; // rbx
  __int64 v10; // rdi
  __int64 v11; // r13
  int v12; // r8d

  v3 = sub_BD5C60(a1, a2, a3);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(_QWORD *)v3;
  result = *(unsigned int *)(*(_QWORD *)v3 + 1736LL);
  v7 = *(_QWORD *)(v5 + 1720);
  if ( (_DWORD)result )
  {
    v8 = result - 1;
    result = ((_DWORD)result - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
    v9 = (__int64 *)(v7 + 16 * result);
    v10 = *v9;
    if ( v4 == *v9 )
    {
LABEL_3:
      v11 = v9[1];
      if ( v11 )
      {
        sub_BD7260(v9[1]);
        result = sub_BD2DD0(v11);
      }
      *v9 = -8192;
      --*(_DWORD *)(v5 + 1728);
      ++*(_DWORD *)(v5 + 1732);
    }
    else
    {
      v12 = 1;
      while ( v10 != -4096 )
      {
        result = v8 & (unsigned int)(v12 + result);
        v9 = (__int64 *)(v7 + 16LL * (unsigned int)result);
        v10 = *v9;
        if ( v4 == *v9 )
          goto LABEL_3;
        ++v12;
      }
    }
  }
  return result;
}
