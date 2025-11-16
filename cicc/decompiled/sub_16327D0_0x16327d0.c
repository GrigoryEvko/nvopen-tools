// Function: sub_16327D0
// Address: 0x16327d0
//
__int64 __fastcall sub_16327D0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r12
  int v4; // r13d
  unsigned int v5; // r15d
  __int64 v6; // rbx
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int32 v10; // [rsp+1Ch] [rbp-54h] BYREF
  __m128i v11; // [rsp+20h] [rbp-50h] BYREF
  __int64 v12; // [rsp+30h] [rbp-40h]

  result = sub_16327A0(a1);
  if ( result )
  {
    v3 = result;
    result = sub_161F520(result);
    v4 = result;
    if ( (_DWORD)result )
    {
      v5 = 0;
      do
      {
        while ( 1 )
        {
          v6 = sub_161F530(v3, v5);
          result = *(unsigned int *)(v6 + 8);
          if ( (unsigned int)result > 2 )
          {
            result = sub_1632720(*(_QWORD *)(v6 - 8 * result), &v10);
            if ( (_BYTE)result )
            {
              v7 = *(unsigned int *)(v6 + 8);
              result = *(_QWORD *)(v6 + 8 * (1 - v7));
              if ( result )
              {
                if ( !*(_BYTE *)result )
                  break;
              }
            }
          }
          if ( v4 == ++v5 )
            return result;
        }
        v11.m128i_i64[1] = *(_QWORD *)(v6 + 8 * (1 - v7));
        v8 = *(unsigned int *)(a2 + 8);
        v12 = *(_QWORD *)(v6 + 8 * (2 - v7));
        v11.m128i_i32[0] = v10;
        if ( (unsigned int)v8 >= *(_DWORD *)(a2 + 12) )
        {
          sub_16CD150(a2, a2 + 16, 0, 24);
          v8 = *(unsigned int *)(a2 + 8);
        }
        ++v5;
        result = *(_QWORD *)a2 + 24 * v8;
        v9 = v12;
        *(__m128i *)result = _mm_loadu_si128(&v11);
        *(_QWORD *)(result + 16) = v9;
        ++*(_DWORD *)(a2 + 8);
      }
      while ( v4 != v5 );
    }
  }
  return result;
}
