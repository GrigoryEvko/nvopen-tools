// Function: sub_2B3C320
// Address: 0x2b3c320
//
__int64 __fastcall sub_2B3C320(__int64 a1)
{
  _QWORD *v2; // rdx
  __int64 v3; // rax
  _QWORD *v4; // rsi
  __int64 v5; // rcx
  __int64 result; // rax
  _QWORD *v7; // rdi
  _QWORD *v8; // r12

  v2 = *(_QWORD **)a1;
  v3 = 8LL * *(unsigned int *)(a1 + 8);
  v4 = (_QWORD *)(*(_QWORD *)a1 + v3);
  v5 = v3 >> 3;
  result = v3 >> 5;
  if ( result )
  {
    v7 = *(_QWORD **)a1;
    result = (__int64)&v2[4 * result];
    while ( *v7 )
    {
      if ( !v7[1] )
      {
        ++v7;
        break;
      }
      if ( !v7[2] )
      {
        v7 += 2;
        break;
      }
      if ( !v7[3] )
      {
        v7 += 3;
        break;
      }
      v7 += 4;
      if ( (_QWORD *)result == v7 )
      {
        v5 = v4 - v7;
        goto LABEL_20;
      }
    }
LABEL_8:
    if ( v4 != v7 )
    {
      result = (__int64)(v7 + 1);
      if ( v4 == v7 + 1 )
      {
        v8 = v7;
      }
      else
      {
        do
        {
          if ( *(_QWORD *)result )
            *v7++ = *(_QWORD *)result;
          result += 8;
        }
        while ( v4 != (_QWORD *)result );
        v2 = *(_QWORD **)a1;
        result = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
        v8 = (_QWORD *)((char *)v7 + result - (_QWORD)v4);
        if ( v4 != (_QWORD *)result )
        {
          result = (__int64)memmove(v7, v4, *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - (_QWORD)v4);
          v2 = *(_QWORD **)a1;
        }
      }
      goto LABEL_15;
    }
LABEL_23:
    v8 = v4;
LABEL_15:
    *(_DWORD *)(a1 + 8) = v8 - v2;
    return result;
  }
  v7 = *(_QWORD **)a1;
LABEL_20:
  if ( v5 != 2 )
  {
    if ( v5 != 3 )
    {
      if ( v5 != 1 )
        goto LABEL_23;
      goto LABEL_28;
    }
    if ( !*v7 )
      goto LABEL_8;
    ++v7;
  }
  if ( !*v7 )
    goto LABEL_8;
  ++v7;
LABEL_28:
  if ( !*v7 )
    goto LABEL_8;
  *(_DWORD *)(a1 + 8) = v4 - v2;
  return result;
}
