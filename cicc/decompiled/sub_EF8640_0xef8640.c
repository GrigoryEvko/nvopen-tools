// Function: sub_EF8640
// Address: 0xef8640
//
void __fastcall sub_EF8640(_QWORD *a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 v3; // rax
  __int64 *v4; // rcx
  unsigned __int64 v5; // rax
  __int64 *v6; // r13
  __int64 v7; // rax
  unsigned __int64 v8; // rax
  __int64 **v9; // rax
  unsigned __int64 v10; // rdx
  size_t v11; // r13
  void *v12; // rax
  void *v13; // rcx

  if ( !*a1 )
  {
    v10 = a1[1];
    if ( v10 == 1 )
    {
      a1[6] = 0;
      v13 = a1 + 6;
    }
    else
    {
      if ( v10 > 0xFFFFFFFFFFFFFFFLL )
        sub_4261EA(a1, a2, v10);
      v11 = 8 * v10;
      v12 = (void *)sub_22077B0(8 * v10);
      v13 = memset(v12, 0, v11);
    }
    *a1 = v13;
  }
  v2 = *(__int64 **)(a2 + 16);
  if ( v2 )
  {
    v3 = sub_22077B0(40);
    v4 = (__int64 *)v3;
    if ( v3 )
      *(_QWORD *)v3 = 0;
    *(__m128i *)(v3 + 8) = _mm_loadu_si128((const __m128i *)(v2 + 1));
    *(_QWORD *)(v3 + 24) = v2[3];
    v5 = v2[4];
    a1[2] = v4;
    v4[4] = v5;
    *(_QWORD *)(*a1 + 8 * (v5 % a1[1])) = a1 + 2;
    while ( 1 )
    {
      v2 = (__int64 *)*v2;
      if ( !v2 )
        break;
      while ( 1 )
      {
        v6 = v4;
        v7 = sub_22077B0(40);
        v4 = (__int64 *)v7;
        if ( v7 )
          *(_QWORD *)v7 = 0;
        *(__m128i *)(v7 + 8) = _mm_loadu_si128((const __m128i *)(v2 + 1));
        *(_QWORD *)(v7 + 24) = v2[3];
        *v6 = v7;
        v8 = v2[4];
        v4[4] = v8;
        v9 = (__int64 **)(*a1 + 8 * (v8 % a1[1]));
        if ( *v9 )
          break;
        *v9 = v6;
        v2 = (__int64 *)*v2;
        if ( !v2 )
          return;
      }
    }
  }
}
