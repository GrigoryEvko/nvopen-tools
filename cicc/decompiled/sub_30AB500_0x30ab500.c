// Function: sub_30AB500
// Address: 0x30ab500
//
__int64 __fastcall sub_30AB500(__int64 *a1, __int64 a2, __int64 a3, unsigned __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 *v7; // r13
  __int64 v8; // rax
  __int64 *v9; // r14
  unsigned __int8 *v10; // rbx
  unsigned __int8 v11; // r12
  const void *v12; // [rsp+0h] [rbp-40h]
  __int64 v13; // [rsp+8h] [rbp-38h]

  v13 = a3;
  result = (unsigned int)*(unsigned __int8 *)a1 - 22;
  if ( (unsigned __int8)(*(_BYTE *)a1 - 22) > 6u )
  {
    v7 = a1;
    v8 = 32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF);
    v9 = &a1[v8 / 0xFFFFFFFFFFFFFFF8LL];
    if ( (*((_BYTE *)a1 + 7) & 0x40) != 0 )
    {
      v9 = (__int64 *)*(a1 - 1);
      v7 = &v9[(unsigned __int64)v8 / 8];
    }
    result = a3 + 16;
    v12 = (const void *)(a3 + 16);
    if ( v9 != v7 )
    {
      while ( 1 )
      {
        v10 = (unsigned __int8 *)*v9;
        if ( *(_BYTE *)(a2 + 28) )
        {
          result = *(_QWORD *)(a2 + 8);
          a4 = *(unsigned int *)(a2 + 20);
          a3 = result + 8 * a4;
          if ( result != a3 )
          {
            while ( v10 != *(unsigned __int8 **)result )
            {
              result += 8;
              if ( a3 == result )
                goto LABEL_20;
            }
            goto LABEL_10;
          }
LABEL_20:
          if ( (unsigned int)a4 < *(_DWORD *)(a2 + 16) )
            break;
        }
        result = (__int64)sub_C8CC70(a2, *v9, a3, a4, a5, a6);
        if ( (_BYTE)a3 )
        {
LABEL_13:
          v11 = *v10;
          if ( *v10 <= 0x1Cu )
            goto LABEL_10;
          result = sub_B46970(v10);
          if ( (_BYTE)result )
            goto LABEL_10;
          a3 = (unsigned int)v11 - 30;
          if ( (unsigned int)a3 <= 0xA )
            goto LABEL_10;
          result = *(unsigned int *)(v13 + 8);
          a4 = *(unsigned int *)(v13 + 12);
          if ( result + 1 > a4 )
          {
            sub_C8D5F0(v13, v12, result + 1, 8u, a5, a6);
            result = *(unsigned int *)(v13 + 8);
          }
          v9 += 4;
          a3 = *(_QWORD *)v13;
          *(_QWORD *)(*(_QWORD *)v13 + 8 * result) = v10;
          ++*(_DWORD *)(v13 + 8);
          if ( v7 == v9 )
            return result;
        }
        else
        {
LABEL_10:
          v9 += 4;
          if ( v7 == v9 )
            return result;
        }
      }
      a4 = (unsigned int)(a4 + 1);
      *(_DWORD *)(a2 + 20) = a4;
      *(_QWORD *)a3 = v10;
      ++*(_QWORD *)a2;
      goto LABEL_13;
    }
  }
  return result;
}
