// Function: sub_2E16DB0
// Address: 0x2e16db0
//
__int64 __fastcall sub_2E16DB0(__int64 a1, int a2)
{
  _DWORD *v3; // rdx
  __int64 v4; // rax
  _DWORD *v5; // r8
  __int64 v6; // rcx
  __int64 result; // rax
  _DWORD *v8; // rdi
  _DWORD *v9; // r12

  v3 = *(_DWORD **)a1;
  v4 = 4LL * *(unsigned int *)(a1 + 8);
  v5 = (_DWORD *)(*(_QWORD *)a1 + v4);
  v6 = v4 >> 2;
  result = v4 >> 4;
  if ( result )
  {
    v8 = *(_DWORD **)a1;
    result = (__int64)&v3[4 * result];
    while ( *v8 != a2 )
    {
      if ( v8[1] == a2 )
      {
        ++v8;
        goto LABEL_8;
      }
      if ( v8[2] == a2 )
      {
        v8 += 2;
        goto LABEL_8;
      }
      if ( v8[3] == a2 )
      {
        v8 += 3;
        goto LABEL_8;
      }
      v8 += 4;
      if ( (_DWORD *)result == v8 )
      {
        v6 = v5 - v8;
        goto LABEL_20;
      }
    }
    goto LABEL_8;
  }
  v8 = *(_DWORD **)a1;
LABEL_20:
  if ( v6 == 2 )
    goto LABEL_26;
  if ( v6 == 3 )
  {
    if ( *v8 == a2 )
      goto LABEL_8;
    ++v8;
LABEL_26:
    if ( *v8 == a2 )
      goto LABEL_8;
    ++v8;
    goto LABEL_28;
  }
  if ( v6 != 1 )
    goto LABEL_23;
LABEL_28:
  v9 = v5;
  if ( *v8 != a2 )
    goto LABEL_15;
LABEL_8:
  if ( v5 == v8 )
  {
LABEL_23:
    v9 = v5;
    goto LABEL_15;
  }
  result = (__int64)(v8 + 1);
  if ( v5 == v8 + 1 )
  {
    v9 = v8;
  }
  else
  {
    do
    {
      if ( *(_DWORD *)result != a2 )
        *v8++ = *(_DWORD *)result;
      result += 4;
    }
    while ( v5 != (_DWORD *)result );
    v3 = *(_DWORD **)a1;
    result = *(_QWORD *)a1 + 4LL * *(unsigned int *)(a1 + 8);
    v9 = (_DWORD *)((char *)v8 + result - (_QWORD)v5);
    if ( v5 != (_DWORD *)result )
    {
      result = (__int64)memmove(v8, v5, *(_QWORD *)a1 + 4LL * *(unsigned int *)(a1 + 8) - (_QWORD)v5);
      v3 = *(_DWORD **)a1;
    }
  }
LABEL_15:
  *(_DWORD *)(a1 + 8) = v9 - v3;
  return result;
}
