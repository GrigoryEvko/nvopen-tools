// Function: sub_2BCF820
// Address: 0x2bcf820
//
__int64 __fastcall sub_2BCF820(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5)
{
  __int64 v5; // r12
  __int64 v8; // rbx
  unsigned int v9; // r13d
  int v10; // edx
  int v11; // r8d
  unsigned int v12; // eax
  unsigned __int8 *v13; // rdi
  unsigned __int8 *v14; // rsi
  int v15; // eax
  __int64 v16; // rcx

  v5 = a2 + 24 * a3;
  if ( a2 != v5 )
  {
    v8 = a2;
    v9 = 0;
    while ( 1 )
    {
      v14 = *(unsigned __int8 **)(v8 + 16);
      if ( *v14 <= 0x1Cu )
        goto LABEL_4;
      v15 = *(_DWORD *)(a4 + 2000);
      v16 = *(_QWORD *)(a4 + 1984);
      if ( v15 )
      {
        v10 = v15 - 1;
        v11 = 1;
        v12 = (v15 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
        v13 = *(unsigned __int8 **)(v16 + 8LL * v12);
        if ( v14 != v13 )
        {
          while ( v13 != (unsigned __int8 *)-4096LL )
          {
            v12 = v10 & (v11 + v12);
            v13 = *(unsigned __int8 **)(v16 + 8LL * v12);
            if ( v14 == v13 )
              goto LABEL_4;
            ++v11;
          }
          goto LABEL_7;
        }
LABEL_4:
        v8 += 24;
        if ( v5 == v8 )
          return v9;
      }
      else
      {
LABEL_7:
        v9 |= sub_2BCF240(a1, v14, a4, a5);
        v8 += 24;
        if ( v5 == v8 )
          return v9;
      }
    }
  }
  return 0;
}
