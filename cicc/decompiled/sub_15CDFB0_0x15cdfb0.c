// Function: sub_15CDFB0
// Address: 0x15cdfb0
//
__int64 __fastcall sub_15CDFB0(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // r8
  _BYTE *v6; // rax
  __int64 v7; // r8
  __int64 result; // rax
  int v9; // edx
  __int64 v10; // rcx
  int v11; // edi
  __int64 *v12; // r13
  __int64 v13; // rsi
  __int64 v14; // rdi
  __int64 v15[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = sub_15CC510(a1, a2);
  *(_BYTE *)(a1 + 72) = 0;
  v5 = *(_QWORD *)(v4 + 8);
  v15[0] = v4;
  if ( v5 )
  {
    v6 = sub_15CBEB0(*(_QWORD **)(v5 + 24), *(_QWORD *)(v5 + 32), v15);
    sub_15CDF70(v7 + 24, v6);
  }
  result = *(unsigned int *)(a1 + 48);
  if ( (_DWORD)result )
  {
    v9 = result - 1;
    v10 = *(_QWORD *)(a1 + 32);
    v11 = 1;
    result = ((_DWORD)result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v12 = (__int64 *)(v10 + 16 * result);
    v13 = *v12;
    if ( *v12 == a2 )
    {
LABEL_5:
      v14 = v12[1];
      if ( v14 )
        result = sub_15CBC60(v14);
      *v12 = -16;
      --*(_DWORD *)(a1 + 40);
      ++*(_DWORD *)(a1 + 44);
    }
    else
    {
      while ( v13 != -8 )
      {
        result = v9 & (unsigned int)(v11 + result);
        v12 = (__int64 *)(v10 + 16LL * (unsigned int)result);
        v13 = *v12;
        if ( *v12 == a2 )
          goto LABEL_5;
        ++v11;
      }
    }
  }
  return result;
}
