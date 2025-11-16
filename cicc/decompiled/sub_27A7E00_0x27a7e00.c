// Function: sub_27A7E00
// Address: 0x27a7e00
//
void __fastcall sub_27A7E00(__int64 a1, int *a2, __int64 a3, __int64 a4)
{
  int *v4; // r13
  char v7; // al
  const __m128i *v8; // rdi
  int v9; // esi
  __int64 v10; // rcx
  __int64 v11; // rax
  __int32 v12; // edx
  __int64 v13; // [rsp+0h] [rbp-30h] BYREF
  __int64 v14; // [rsp+8h] [rbp-28h]

  v13 = a3;
  v14 = a4;
  if ( (int *)a1 != a2 )
  {
    v4 = (int *)(a1 + 16);
    if ( (int *)(a1 + 16) != a2 )
    {
      do
      {
        v7 = sub_27A2220(&v13, v4, a1);
        v8 = (const __m128i *)v4;
        v4 += 4;
        if ( v7 )
        {
          v9 = *(v4 - 4);
          v10 = *((_QWORD *)v4 - 1);
          v11 = ((__int64)v8->m128i_i64 - a1) >> 4;
          if ( (__int64)v8->m128i_i64 - a1 > 0 )
          {
            do
            {
              v12 = v8[-1].m128i_i32[0];
              --v8;
              v8[1].m128i_i32[0] = v12;
              v8[1].m128i_i64[1] = v8->m128i_i64[1];
              --v11;
            }
            while ( v11 );
          }
          *(_DWORD *)a1 = v9;
          *(_QWORD *)(a1 + 8) = v10;
        }
        else
        {
          sub_27A7D80(v8, v13, v14);
        }
      }
      while ( a2 != v4 );
    }
  }
}
