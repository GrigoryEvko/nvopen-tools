// Function: sub_2A8A7B0
// Address: 0x2a8a7b0
//
__int64 __fastcall sub_2A8A7B0(__int64 a1, __int64 *a2, _QWORD *a3)
{
  __int64 result; // rax
  int v5; // edx
  _QWORD *v6; // r11
  int v7; // r12d
  __int64 v8; // rcx
  __int64 v9; // r10
  __int64 v10; // rsi
  unsigned int i; // eax
  _QWORD *v12; // rdi
  __int64 v13; // r9
  bool v14; // bl
  unsigned int v15; // eax

  result = *(unsigned int *)(a1 + 24);
  if ( !(_DWORD)result )
  {
    *a3 = 0;
    return result;
  }
  v5 = result - 1;
  v6 = 0;
  v7 = 1;
  v8 = *a2;
  v9 = *(_QWORD *)(a1 + 8);
  v10 = *(_QWORD *)(*a2 + 16);
  for ( i = (result - 1) & (((unsigned int)v10 >> 4) ^ ((unsigned int)v10 >> 9)); ; i = v5 & v15 )
  {
    v12 = (_QWORD *)(v9 + 8LL * i);
    v13 = *v12;
    if ( v8 == -4096 )
    {
      if ( v13 == -4096 )
        goto LABEL_8;
      v14 = v13 == -8192;
      goto LABEL_12;
    }
    if ( v13 == -4096 )
      break;
    v14 = v13 == -8192;
    if ( v8 == -8192 )
    {
      if ( v13 == -8192 )
      {
LABEL_8:
        *a3 = v12;
        return 1;
      }
    }
    else if ( v13 != -8192 )
    {
      if ( v10 == *(_QWORD *)(v13 + 16) )
        goto LABEL_8;
LABEL_17:
      v12 = v6;
      goto LABEL_14;
    }
LABEL_12:
    if ( v6 || !v14 )
      goto LABEL_17;
LABEL_14:
    v15 = v7 + i;
    v6 = v12;
    ++v7;
  }
  if ( !v6 )
    v6 = (_QWORD *)(v9 + 8LL * i);
  *a3 = v6;
  return 0;
}
