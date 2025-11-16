// Function: sub_C50AB0
// Address: 0xc50ab0
//
__int64 __fastcall sub_C50AB0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 result; // rax
  __int64 v5; // rcx
  _QWORD *v6; // rdx
  __int64 *v7; // r13
  __int64 *v8; // r14
  char v9; // di
  __int64 v10; // rsi
  __int64 v11; // rcx
  _QWORD *v12; // rdx

  v3 = *a1;
  if ( *a1 )
  {
    if ( !*(_BYTE *)(a2 + 124) )
      return sub_C8CC70(a2 + 96, v3);
    result = *(_QWORD *)(a2 + 104);
    v5 = *(unsigned int *)(a2 + 116);
    v6 = (_QWORD *)(result + 8 * v5);
    if ( (_QWORD *)result == v6 )
    {
LABEL_23:
      if ( (unsigned int)v5 < *(_DWORD *)(a2 + 112) )
      {
        *(_DWORD *)(a2 + 116) = v5 + 1;
        *v6 = v3;
        ++*(_QWORD *)(a2 + 96);
        return result;
      }
      return sub_C8CC70(a2 + 96, v3);
    }
    while ( v3 != *(_QWORD *)result )
    {
      result += 8;
      if ( v6 == (_QWORD *)result )
        goto LABEL_23;
    }
  }
  else
  {
    result = a1[1];
    if ( result )
    {
      v7 = *(__int64 **)result;
      result = *(unsigned int *)(result + 8);
      v8 = &v7[result];
      if ( v8 != v7 )
      {
        v9 = *(_BYTE *)(a2 + 124);
        v10 = *v7;
        if ( !v9 )
          goto LABEL_18;
LABEL_12:
        result = *(_QWORD *)(a2 + 104);
        v11 = *(unsigned int *)(a2 + 116);
        v12 = (_QWORD *)(result + 8 * v11);
        if ( (_QWORD *)result == v12 )
        {
LABEL_20:
          if ( (unsigned int)v11 >= *(_DWORD *)(a2 + 112) )
          {
LABEL_18:
            while ( 1 )
            {
              ++v7;
              result = sub_C8CC70(a2 + 96, v10);
              v9 = *(_BYTE *)(a2 + 124);
              if ( v8 == v7 )
                break;
LABEL_17:
              v10 = *v7;
              if ( v9 )
                goto LABEL_12;
            }
          }
          else
          {
            ++v7;
            *(_DWORD *)(a2 + 116) = v11 + 1;
            *v12 = v10;
            v9 = *(_BYTE *)(a2 + 124);
            ++*(_QWORD *)(a2 + 96);
            if ( v8 != v7 )
              goto LABEL_17;
          }
        }
        else
        {
          while ( v10 != *(_QWORD *)result )
          {
            result += 8;
            if ( v12 == (_QWORD *)result )
              goto LABEL_20;
          }
          if ( v8 != ++v7 )
            goto LABEL_17;
        }
      }
    }
  }
  return result;
}
