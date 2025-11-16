// Function: sub_1D46770
// Address: 0x1d46770
//
__int64 __fastcall sub_1D46770(_QWORD *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 v6; // rbx
  __int64 v7; // r12
  _QWORD *v8; // rdx
  __int64 v9; // rax
  _QWORD *v10; // rsi
  __int64 v11; // rcx
  __int64 result; // rax
  _QWORD *v13; // rdi
  _QWORD *v14; // r13

  v6 = *a2;
  sub_1D455B0(*(_QWORD *)(*a1 + 272LL), *a2, a3, a4, a5, a6);
  v7 = a1[1];
  v8 = *(_QWORD **)v7;
  v9 = 8LL * *(unsigned int *)(v7 + 8);
  v10 = (_QWORD *)(*(_QWORD *)v7 + v9);
  v11 = v9 >> 3;
  result = v9 >> 5;
  if ( result )
  {
    v13 = *(_QWORD **)v7;
    result = (__int64)&v8[4 * result];
    while ( v6 != *v13 )
    {
      if ( v6 == v13[1] )
      {
        ++v13;
        goto LABEL_8;
      }
      if ( v6 == v13[2] )
      {
        v13 += 2;
        goto LABEL_8;
      }
      if ( v6 == v13[3] )
      {
        v13 += 3;
        goto LABEL_8;
      }
      v13 += 4;
      if ( (_QWORD *)result == v13 )
      {
        v11 = v10 - v13;
        goto LABEL_20;
      }
    }
    goto LABEL_8;
  }
  v13 = *(_QWORD **)v7;
LABEL_20:
  if ( v11 == 2 )
    goto LABEL_26;
  if ( v11 == 3 )
  {
    if ( v6 == *v13 )
      goto LABEL_8;
    ++v13;
LABEL_26:
    if ( v6 == *v13 )
      goto LABEL_8;
    ++v13;
    goto LABEL_28;
  }
  if ( v11 != 1 )
    goto LABEL_23;
LABEL_28:
  v14 = v10;
  if ( v6 != *v13 )
    goto LABEL_15;
LABEL_8:
  if ( v10 == v13 )
  {
LABEL_23:
    v14 = v10;
    goto LABEL_15;
  }
  result = (__int64)(v13 + 1);
  if ( v10 == v13 + 1 )
  {
    v14 = v13;
  }
  else
  {
    do
    {
      if ( v6 != *(_QWORD *)result )
        *v13++ = *(_QWORD *)result;
      result += 8;
    }
    while ( v10 != (_QWORD *)result );
    v8 = *(_QWORD **)v7;
    result = *(_QWORD *)v7 + 8LL * *(unsigned int *)(v7 + 8);
    v14 = (_QWORD *)((char *)v13 + result - (_QWORD)v10);
    if ( v10 != (_QWORD *)result )
    {
      result = (__int64)memmove(v13, v10, *(_QWORD *)v7 + 8LL * *(unsigned int *)(v7 + 8) - (_QWORD)v10);
      v8 = *(_QWORD **)v7;
    }
  }
LABEL_15:
  *(_DWORD *)(v7 + 8) = v14 - v8;
  return result;
}
