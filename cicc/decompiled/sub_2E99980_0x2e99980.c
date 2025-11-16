// Function: sub_2E99980
// Address: 0x2e99980
//
__int64 *__fastcall sub_2E99980(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 *result; // rax
  _QWORD *v5; // rbx
  __int64 v6; // r13
  __int64 v7; // rbx
  _BYTE *v8; // r12
  __int64 v9; // rax
  _BYTE *v10; // r15
  _BYTE *v11; // rbx
  unsigned int v12; // edx
  _BYTE *v13; // r15
  __m128i *v14; // rsi
  __int64 *v15; // [rsp+8h] [rbp-78h]
  __int64 *v16; // [rsp+10h] [rbp-70h]
  _QWORD *j; // [rsp+18h] [rbp-68h]
  __int32 v18; // [rsp+20h] [rbp-60h]
  unsigned int i; // [rsp+24h] [rbp-5Ch]
  __m128i v20; // [rsp+30h] [rbp-50h] BYREF
  __int64 v21; // [rsp+40h] [rbp-40h]

  result = *(__int64 **)(a3 + 32);
  v18 = (unsigned __int16)a2;
  v15 = *(__int64 **)(a3 + 40);
  v16 = result;
  for ( i = a2 - 1; v15 != v16; result = v16 )
  {
    v5 = (_QWORD *)*v16;
    if ( !(unsigned __int8)sub_2E31DD0(*v16, a2, -1, -1) )
    {
      v20.m128i_i64[1] = -1;
      v21 = -1;
      v14 = (__m128i *)v5[24];
      v20.m128i_i32[0] = v18;
      if ( v14 == (__m128i *)v5[25] )
      {
        sub_2E341F0(v5 + 23, v14, &v20);
      }
      else
      {
        if ( v14 )
        {
          *v14 = _mm_loadu_si128(&v20);
          v14[1].m128i_i64[0] = v21;
          v14 = (__m128i *)v5[24];
        }
        v5[24] = (char *)v14 + 24;
      }
    }
    v6 = v5[7];
    for ( j = v5 + 6; (_QWORD *)v6 != j; v6 = *(_QWORD *)(v6 + 8) )
    {
      v7 = *(_QWORD *)(v6 + 32);
      v8 = (_BYTE *)(v7 + 40LL * (*(_DWORD *)(v6 + 40) & 0xFFFFFF));
      v9 = 5LL * (unsigned int)sub_2E88FE0(v6);
      if ( v8 != (_BYTE *)(v7 + 8 * v9) )
      {
        v10 = (_BYTE *)(v7 + 8 * v9);
        while ( 1 )
        {
          v11 = v10;
          if ( (unsigned __int8)sub_2E2FA70(v10) )
            break;
          v10 += 40;
          if ( v8 == v10 )
            goto LABEL_18;
        }
        if ( v10 != v8 )
        {
          do
          {
            v12 = *((_DWORD *)v11 + 2);
            if ( v12
              && (v12 == a2
               || i <= 0x3FFFFFFE && v12 - 1 <= 0x3FFFFFFE && (unsigned __int8)sub_E92070(*(_QWORD *)(a1 + 16), a2, v12)) )
            {
              v11[3] &= ~0x40u;
            }
            if ( v11 + 40 == v8 )
              break;
            v13 = v11 + 40;
            while ( 1 )
            {
              v11 = v13;
              if ( (unsigned __int8)sub_2E2FA70(v13) )
                break;
              v13 += 40;
              if ( v8 == v13 )
                goto LABEL_18;
            }
          }
          while ( v8 != v13 );
        }
      }
LABEL_18:
      if ( (*(_BYTE *)v6 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v6 + 44) & 8) != 0 )
          v6 = *(_QWORD *)(v6 + 8);
      }
    }
    ++v16;
  }
  return result;
}
