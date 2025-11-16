// Function: sub_C605E0
// Address: 0xc605e0
//
void __fastcall sub_C605E0(__int64 a1, const __m128i *a2, __int64 a3)
{
  __int64 v3; // rdx
  const __m128i *v4; // r13
  const __m128i *v5; // rbx
  _BYTE *v6; // rax
  __int64 v7; // rdx
  _OWORD v8[3]; // [rsp+0h] [rbp-30h] BYREF

  if ( a3 )
  {
    v3 = a3;
    v4 = &a2[v3];
    if ( a2 != &a2[v3] )
    {
      v5 = a2 + 1;
      v8[0] = _mm_loadu_si128(a2);
      sub_C60570((__int64 *)v8, a1);
      if ( v4 != &a2[1] )
      {
        do
        {
          v6 = *(_BYTE **)(a1 + 32);
          v8[0] = _mm_loadu_si128(v5);
          if ( (unsigned __int64)v6 >= *(_QWORD *)(a1 + 24) )
          {
            sub_CB5D20(a1, 58);
          }
          else
          {
            *(_QWORD *)(a1 + 32) = v6 + 1;
            *v6 = 58;
          }
          ++v5;
          sub_C60570((__int64 *)v8, a1);
        }
        while ( v4 != v5 );
      }
    }
  }
  else
  {
    v7 = *(_QWORD *)(a1 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(a1 + 24) - v7) <= 4 )
    {
      sub_CB6200(a1, "empty", 5);
    }
    else
    {
      *(_DWORD *)v7 = 1953525093;
      *(_BYTE *)(v7 + 4) = 121;
      *(_QWORD *)(a1 + 32) += 5LL;
    }
  }
}
