// Function: sub_C578B0
// Address: 0xc578b0
//
__int64 __fastcall sub_C578B0(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 v5; // rsi
  __int64 result; // rax
  __int64 v7; // rcx
  _QWORD *v8; // rdx
  __int64 *v9; // r13
  __int64 *v10; // r14
  char v11; // di
  __int64 v12; // rsi
  __int64 v13; // rcx
  _QWORD *v14; // rdx

  sub_C57500(a1, a2);
  v5 = *a3;
  if ( *a3 )
  {
    if ( !*(_BYTE *)(a1 + 124) )
      return sub_C8CC70(a1 + 96, v5);
    result = *(_QWORD *)(a1 + 104);
    v7 = *(unsigned int *)(a1 + 116);
    v8 = (_QWORD *)(result + 8 * v7);
    if ( (_QWORD *)result == v8 )
    {
LABEL_23:
      if ( (unsigned int)v7 < *(_DWORD *)(a1 + 112) )
      {
        *(_DWORD *)(a1 + 116) = v7 + 1;
        *v8 = v5;
        ++*(_QWORD *)(a1 + 96);
        return result;
      }
      return sub_C8CC70(a1 + 96, v5);
    }
    while ( v5 != *(_QWORD *)result )
    {
      result += 8;
      if ( v8 == (_QWORD *)result )
        goto LABEL_23;
    }
  }
  else
  {
    result = a3[1];
    if ( result )
    {
      v9 = *(__int64 **)result;
      result = *(unsigned int *)(result + 8);
      v10 = &v9[result];
      if ( v9 != v10 )
      {
        v11 = *(_BYTE *)(a1 + 124);
        v12 = *v9;
        if ( !v11 )
          goto LABEL_18;
LABEL_12:
        result = *(_QWORD *)(a1 + 104);
        v13 = *(unsigned int *)(a1 + 116);
        v14 = (_QWORD *)(result + 8 * v13);
        if ( (_QWORD *)result == v14 )
        {
LABEL_20:
          if ( (unsigned int)v13 >= *(_DWORD *)(a1 + 112) )
          {
LABEL_18:
            while ( 1 )
            {
              ++v9;
              result = sub_C8CC70(a1 + 96, v12);
              v11 = *(_BYTE *)(a1 + 124);
              if ( v10 == v9 )
                break;
LABEL_17:
              v12 = *v9;
              if ( v11 )
                goto LABEL_12;
            }
          }
          else
          {
            ++v9;
            *(_DWORD *)(a1 + 116) = v13 + 1;
            *v14 = v12;
            v11 = *(_BYTE *)(a1 + 124);
            ++*(_QWORD *)(a1 + 96);
            if ( v10 != v9 )
              goto LABEL_17;
          }
        }
        else
        {
          while ( v12 != *(_QWORD *)result )
          {
            result += 8;
            if ( v14 == (_QWORD *)result )
              goto LABEL_20;
          }
          if ( v10 != ++v9 )
            goto LABEL_17;
        }
      }
    }
  }
  return result;
}
