// Function: sub_D682A0
// Address: 0xd682a0
//
__int64 __fastcall sub_D682A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdx
  __int64 v6; // r11
  unsigned int v7; // r9d
  __int64 v8; // rax
  unsigned int v9; // esi
  __int64 v10; // rax
  __int64 result; // rax
  _QWORD *v12; // r10
  _QWORD *i; // rdx
  __int64 v14; // r9
  __int64 v15; // r9

  v5 = *(_QWORD *)(a1 - 8);
  v6 = 32LL * *(unsigned int *)(a1 + 76);
  v7 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  if ( v7 )
  {
    v8 = 0;
    while ( a2 != *(_QWORD *)(v5 + v6 + 8 * v8) )
    {
      if ( v7 == (_DWORD)++v8 )
        goto LABEL_6;
    }
    v9 = v8;
    v10 = 8LL * (int)v8;
  }
  else
  {
LABEL_6:
    v10 = -8;
    v9 = -1;
  }
  result = v6 + v10;
  v12 = (_QWORD *)(v5 + v6 + 8LL * v7);
  for ( i = (_QWORD *)(result + v5); v12 != i; ++v9 )
  {
    if ( a2 != *i )
      break;
    result = *(_QWORD *)(a1 - 8) + 32LL * v9;
    if ( *(_QWORD *)result )
    {
      v14 = *(_QWORD *)(result + 8);
      **(_QWORD **)(result + 16) = v14;
      if ( v14 )
        *(_QWORD *)(v14 + 16) = *(_QWORD *)(result + 16);
    }
    *(_QWORD *)result = a3;
    if ( a3 )
    {
      v15 = *(_QWORD *)(a3 + 16);
      *(_QWORD *)(result + 8) = v15;
      if ( v15 )
        *(_QWORD *)(v15 + 16) = result + 8;
      *(_QWORD *)(result + 16) = a3 + 16;
      *(_QWORD *)(a3 + 16) = result;
    }
    ++i;
  }
  return result;
}
