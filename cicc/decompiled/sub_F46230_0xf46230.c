// Function: sub_F46230
// Address: 0xf46230
//
__int64 __fastcall sub_F46230(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  __int64 v6; // r12
  __int64 v7; // r15
  __int64 v9; // rbx
  __int64 i; // r14
  __int64 v11; // r9
  const void *v12; // [rsp+0h] [rbp-40h]
  __int64 v13; // [rsp+8h] [rbp-38h]

  result = a3 + 16;
  v6 = a1 + 8 * a2;
  v12 = (const void *)(a3 + 16);
  if ( v6 != a1 )
  {
    v7 = a1;
    do
    {
      v9 = *(_QWORD *)(*(_QWORD *)v7 + 56LL);
      for ( i = *(_QWORD *)v7 + 48LL; i != v9; v9 = *(_QWORD *)(v9 + 8) )
      {
        while ( 1 )
        {
          if ( !v9 )
            BUG();
          if ( *(_BYTE *)(v9 - 24) == 85 )
          {
            result = *(_QWORD *)(v9 - 56);
            if ( result )
            {
              if ( !*(_BYTE *)result
                && *(_QWORD *)(result + 24) == *(_QWORD *)(v9 + 56)
                && (*(_BYTE *)(result + 33) & 0x20) != 0
                && *(_DWORD *)(result + 36) == 155 )
              {
                break;
              }
            }
          }
          v9 = *(_QWORD *)(v9 + 8);
          if ( i == v9 )
            goto LABEL_16;
        }
        v11 = *(_QWORD *)(*(_QWORD *)(v9 - 32LL * (*(_DWORD *)(v9 - 20) & 0x7FFFFFF) - 24) + 24LL);
        result = *(unsigned int *)(a3 + 8);
        if ( result + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
        {
          v13 = *(_QWORD *)(*(_QWORD *)(v9 - 32LL * (*(_DWORD *)(v9 - 20) & 0x7FFFFFF) - 24) + 24LL);
          sub_C8D5F0(a3, v12, result + 1, 8u, a5, v11);
          result = *(unsigned int *)(a3 + 8);
          v11 = v13;
        }
        *(_QWORD *)(*(_QWORD *)a3 + 8 * result) = v11;
        ++*(_DWORD *)(a3 + 8);
      }
LABEL_16:
      v7 += 8;
    }
    while ( v6 != v7 );
  }
  return result;
}
