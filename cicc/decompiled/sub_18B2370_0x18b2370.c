// Function: sub_18B2370
// Address: 0x18b2370
//
__int64 __fastcall sub_18B2370(__int64 a1, __int64 a2)
{
  __int64 *v3; // rax
  __int64 v4; // r14
  __int64 result; // rax
  __int64 v6; // r12
  __int64 i; // r13
  __int64 v8; // rsi
  __int64 *v9; // rdi
  unsigned int v10; // r8d
  _QWORD *v11; // rcx
  __int64 v12; // rdx
  __int64 *v13; // rsi
  __int64 *v14; // rcx

  v3 = *(__int64 **)(a2 + 8);
  if ( *(__int64 **)(a2 + 16) != v3 )
    goto LABEL_2;
  v12 = *(unsigned int *)(a2 + 28);
  v13 = &v3[v12];
  if ( v3 == v13 )
  {
LABEL_29:
    if ( (unsigned int)v12 >= *(_DWORD *)(a2 + 24) )
    {
LABEL_2:
      sub_16CCBA0(a2, a1);
      goto LABEL_3;
    }
    *(_DWORD *)(a2 + 28) = v12 + 1;
    *v13 = a1;
    ++*(_QWORD *)a2;
  }
  else
  {
    v14 = 0;
    while ( a1 != *v3 )
    {
      if ( *v3 == -2 )
        v14 = v3;
      if ( v13 == ++v3 )
      {
        if ( !v14 )
          goto LABEL_29;
        *v14 = a1;
        --*(_DWORD *)(a2 + 32);
        ++*(_QWORD *)a2;
        break;
      }
    }
  }
LABEL_3:
  v4 = *(_QWORD *)(a1 - 24);
  result = *(_DWORD *)(v4 + 20) & 0xFFFFFFF;
  if ( (*(_DWORD *)(v4 + 20) & 0xFFFFFFF) != 0 )
  {
    v6 = (unsigned int)(result - 1);
    for ( i = 0; ; ++i )
    {
      result = sub_1649C60(*(_QWORD *)(v4 + 24 * (i - (unsigned int)result)));
      v8 = result;
      if ( *(_BYTE *)(result + 16) <= 3u )
      {
        result = *(_QWORD *)(a2 + 8);
        if ( *(_QWORD *)(a2 + 16) != result )
          goto LABEL_5;
        v9 = (__int64 *)(result + 8LL * *(unsigned int *)(a2 + 28));
        v10 = *(_DWORD *)(a2 + 28);
        if ( (__int64 *)result == v9 )
        {
LABEL_27:
          if ( v10 >= *(_DWORD *)(a2 + 24) )
          {
LABEL_5:
            result = (__int64)sub_16CCBA0(a2, v8);
            goto LABEL_6;
          }
          *(_DWORD *)(a2 + 28) = v10 + 1;
          *v9 = v8;
          ++*(_QWORD *)a2;
        }
        else
        {
          v11 = 0;
          while ( v8 != *(_QWORD *)result )
          {
            if ( *(_QWORD *)result == -2 )
              v11 = (_QWORD *)result;
            result += 8;
            if ( v9 == (__int64 *)result )
            {
              if ( !v11 )
                goto LABEL_27;
              *v11 = v8;
              --*(_DWORD *)(a2 + 32);
              ++*(_QWORD *)a2;
              if ( i != v6 )
                goto LABEL_7;
              return result;
            }
          }
        }
      }
LABEL_6:
      if ( i == v6 )
        return result;
LABEL_7:
      LODWORD(result) = *(_DWORD *)(v4 + 20) & 0xFFFFFFF;
    }
  }
  return result;
}
