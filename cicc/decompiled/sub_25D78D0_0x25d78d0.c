// Function: sub_25D78D0
// Address: 0x25d78d0
//
unsigned __int64 __fastcall sub_25D78D0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  unsigned __int64 result; // rax
  char v9; // dl
  __int64 v10; // rax
  unsigned __int64 v11; // r8
  unsigned __int64 v12; // r9
  unsigned __int64 v13; // rdx
  _QWORD *v14; // rdi
  unsigned __int64 v15; // r10
  _QWORD *v16; // rdx
  _QWORD *v17; // rsi
  _QWORD *v18; // r12
  _QWORD *v19; // r14

  v6 = (__int64)a3;
  if ( !*(_BYTE *)(a1 + 36) )
    goto LABEL_8;
  result = *(_QWORD *)(a1 + 16);
  a4 = *(unsigned int *)(a1 + 28);
  a3 = (__int64 *)(result + 8 * a4);
  if ( (__int64 *)result == a3 )
  {
LABEL_7:
    if ( (unsigned int)a4 < *(_DWORD *)(a1 + 24) )
    {
      *(_DWORD *)(a1 + 28) = a4 + 1;
      *a3 = a2;
      ++*(_QWORD *)(a1 + 8);
      if ( !v6 )
        goto LABEL_13;
      goto LABEL_10;
    }
LABEL_8:
    result = (unsigned __int64)sub_C8CC70(a1 + 8, a2, (__int64)a3, a4, a5, a6);
    if ( !v9 )
      return result;
    if ( !v6 )
      goto LABEL_13;
LABEL_10:
    v10 = *(unsigned int *)(v6 + 8);
    if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(v6 + 12) )
    {
      sub_C8D5F0(v6, (const void *)(v6 + 16), v10 + 1, 8u, a5, a6);
      v10 = *(unsigned int *)(v6 + 8);
    }
    *(_QWORD *)(*(_QWORD *)v6 + 8 * v10) = a2;
    ++*(_DWORD *)(v6 + 8);
LABEL_13:
    result = sub_B326A0(a2);
    v11 = result;
    if ( result )
    {
      v12 = *(_QWORD *)(a1 + 392);
      v13 = result % v12;
      result = *(_QWORD *)(a1 + 384);
      v14 = *(_QWORD **)(result + 8 * v13);
      v15 = v13;
      if ( v14 )
      {
        v16 = (_QWORD *)*v14;
        if ( v11 == *(_QWORD *)(*v14 + 8LL) )
        {
LABEL_19:
          v18 = (_QWORD *)*v14;
          if ( *v14 )
          {
            v19 = (_QWORD *)*v18;
            if ( !*v18 )
              goto LABEL_32;
            while ( 1 )
            {
              result = v19[1] / v12;
              if ( v19[1] % v12 != v15 || v11 != v19[1] )
                break;
              v19 = (_QWORD *)*v19;
              if ( !v19 )
                goto LABEL_24;
            }
            if ( v18 != v19 )
            {
LABEL_32:
              do
              {
LABEL_24:
                result = sub_25D78D0(a1, v18[2], v6);
                v18 = (_QWORD *)*v18;
              }
              while ( v19 != v18 );
            }
          }
        }
        else
        {
          while ( 1 )
          {
            v17 = (_QWORD *)*v16;
            if ( !*v16 )
              break;
            v14 = v16;
            result = v17[1] / v12;
            if ( v15 != v17[1] % v12 )
              break;
            v16 = (_QWORD *)*v16;
            if ( v11 == v17[1] )
              goto LABEL_19;
          }
        }
      }
    }
    return result;
  }
  while ( a2 != *(_QWORD *)result )
  {
    result += 8LL;
    if ( a3 == (__int64 *)result )
      goto LABEL_7;
  }
  return result;
}
