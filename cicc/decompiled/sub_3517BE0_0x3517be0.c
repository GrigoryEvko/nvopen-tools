// Function: sub_3517BE0
// Address: 0x3517be0
//
__int64 __fastcall sub_3517BE0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rdx
  __int64 v4; // rax
  _QWORD *v5; // r8
  __int64 v6; // rcx
  __int64 result; // rax
  _QWORD *v8; // rdi
  _QWORD *v9; // r12

  v3 = *(_QWORD **)a1;
  v4 = 8LL * *(unsigned int *)(a1 + 8);
  v5 = (_QWORD *)(*(_QWORD *)a1 + v4);
  v6 = v4 >> 3;
  result = v4 >> 5;
  if ( result )
  {
    v8 = *(_QWORD **)a1;
    result = (__int64)&v3[4 * result];
    while ( *v8 != a2 )
    {
      if ( v8[1] == a2 )
      {
        ++v8;
        break;
      }
      if ( v8[2] == a2 )
      {
        v8 += 2;
        break;
      }
      if ( v8[3] == a2 )
      {
        v8 += 3;
        break;
      }
      v8 += 4;
      if ( (_QWORD *)result == v8 )
      {
        v6 = v5 - v8;
        goto LABEL_20;
      }
    }
LABEL_8:
    if ( v5 != v8 )
    {
      result = (__int64)(v8 + 1);
      if ( v5 == v8 + 1 )
      {
        v9 = v8;
      }
      else
      {
        do
        {
          if ( *(_QWORD *)result != a2 )
            *v8++ = *(_QWORD *)result;
          result += 8;
        }
        while ( v5 != (_QWORD *)result );
        v3 = *(_QWORD **)a1;
        result = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
        v9 = (_QWORD *)((char *)v8 + result - (_QWORD)v5);
        if ( v5 != (_QWORD *)result )
        {
          result = (__int64)memmove(v8, v5, *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - (_QWORD)v5);
          v3 = *(_QWORD **)a1;
        }
      }
      goto LABEL_15;
    }
LABEL_23:
    v9 = v5;
LABEL_15:
    *(_DWORD *)(a1 + 8) = v9 - v3;
    return result;
  }
  v8 = *(_QWORD **)a1;
LABEL_20:
  if ( v6 != 2 )
  {
    if ( v6 != 3 )
    {
      if ( v6 != 1 )
        goto LABEL_23;
      goto LABEL_28;
    }
    if ( *v8 == a2 )
      goto LABEL_8;
    ++v8;
  }
  if ( *v8 == a2 )
    goto LABEL_8;
  ++v8;
LABEL_28:
  if ( *v8 == a2 )
    goto LABEL_8;
  *(_DWORD *)(a1 + 8) = v5 - v3;
  return result;
}
