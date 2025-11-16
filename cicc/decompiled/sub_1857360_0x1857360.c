// Function: sub_1857360
// Address: 0x1857360
//
unsigned __int64 __fastcall sub_1857360(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  unsigned __int64 result; // rax
  char v10; // dl
  __int64 v11; // rax
  unsigned __int64 v12; // rcx
  unsigned __int64 v13; // r9
  unsigned __int64 v14; // rdx
  unsigned __int64 *v15; // r8
  unsigned __int64 v16; // rdi
  _QWORD *v17; // r12
  _QWORD *v18; // r14
  __int64 *v19; // rsi
  unsigned int v20; // edi
  _QWORD *v21; // rcx

  result = *(_QWORD *)(a1 + 8);
  if ( *(_QWORD *)(a1 + 16) != result )
    goto LABEL_2;
  v19 = (__int64 *)(result + 8LL * *(unsigned int *)(a1 + 28));
  v20 = *(_DWORD *)(a1 + 28);
  if ( (__int64 *)result == v19 )
  {
LABEL_32:
    if ( v20 < *(_DWORD *)(a1 + 24) )
    {
      *(_DWORD *)(a1 + 28) = v20 + 1;
      *v19 = a2;
      ++*(_QWORD *)a1;
LABEL_3:
      if ( a3 )
      {
LABEL_4:
        v11 = *(unsigned int *)(a3 + 8);
        if ( (unsigned int)v11 >= *(_DWORD *)(a3 + 12) )
        {
          sub_16CD150(a3, (const void *)(a3 + 16), 0, 8, a5, a6);
          v11 = *(unsigned int *)(a3 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a3 + 8 * v11) = a2;
        ++*(_DWORD *)(a3 + 8);
      }
LABEL_7:
      result = sub_15E4F10(a2);
      v12 = result;
      if ( result )
      {
        v13 = *(_QWORD *)(a1 + 392);
        v14 = result % v13;
        result = *(_QWORD *)(a1 + 384);
        v15 = *(unsigned __int64 **)(result + 8 * v14);
        if ( v15 )
        {
          result = *v15;
          if ( v12 == *(_QWORD *)(*v15 + 8) )
          {
LABEL_13:
            v17 = (_QWORD *)*v15;
            if ( *v15 )
            {
              v18 = (_QWORD *)*v17;
              if ( !*v17 )
                goto LABEL_36;
              while ( 1 )
              {
                result = v18[1] / v13;
                if ( v18[1] % v13 != v14 || v12 != v18[1] )
                  break;
                v18 = (_QWORD *)*v18;
                if ( !v18 )
                  goto LABEL_18;
              }
              if ( v17 != v18 )
              {
LABEL_36:
                do
                {
LABEL_18:
                  result = sub_1857360(a1, v17[2], a3);
                  v17 = (_QWORD *)*v17;
                }
                while ( v18 != v17 );
              }
            }
          }
          else
          {
            while ( 1 )
            {
              v16 = *(_QWORD *)result;
              if ( !*(_QWORD *)result )
                break;
              v15 = (unsigned __int64 *)result;
              result = *(_QWORD *)(v16 + 8) / v13;
              if ( v14 != *(_QWORD *)(v16 + 8) % v13 )
                break;
              result = v16;
              if ( v12 == *(_QWORD *)(v16 + 8) )
                goto LABEL_13;
            }
          }
        }
      }
      return result;
    }
LABEL_2:
    result = (unsigned __int64)sub_16CCBA0(a1, a2);
    if ( !v10 )
      return result;
    goto LABEL_3;
  }
  v21 = 0;
  while ( a2 != *(_QWORD *)result )
  {
    if ( *(_QWORD *)result == -2 )
      v21 = (_QWORD *)result;
    result += 8LL;
    if ( v19 == (__int64 *)result )
    {
      if ( !v21 )
        goto LABEL_32;
      *v21 = a2;
      --*(_DWORD *)(a1 + 32);
      ++*(_QWORD *)a1;
      if ( a3 )
        goto LABEL_4;
      goto LABEL_7;
    }
  }
  return result;
}
