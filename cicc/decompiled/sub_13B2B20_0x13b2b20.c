// Function: sub_13B2B20
// Address: 0x13b2b20
//
__int64 __fastcall sub_13B2B20(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r12
  __int64 v4; // r14
  __int64 j; // r15
  __int64 v6; // r13
  __int64 i; // r12
  char v8; // al
  __int64 v9; // r15
  void *v10; // rdx
  __int64 v11; // rdx
  unsigned int v12; // ebx
  __m128i *v13; // rdx
  __m128i si128; // xmm0
  __int64 v15; // rdi
  void *v16; // rdx
  __int64 v17; // r14
  size_t v18; // rax
  _WORD *v19; // rdx
  __int64 v20; // [rsp+10h] [rbp-70h]
  __int64 v21; // [rsp+20h] [rbp-60h]
  __int64 v22; // [rsp+28h] [rbp-58h]
  __int64 v23; // [rsp+30h] [rbp-50h]
  __int64 v24; // [rsp+38h] [rbp-48h]
  __m128i *v25; // [rsp+48h] [rbp-38h] BYREF

  v22 = *(_QWORD *)(a1 + 160);
  result = *(_QWORD *)(v22 + 24);
  v3 = *(_QWORD *)(result + 80);
  v4 = result + 72;
  if ( result + 72 != v3 )
  {
    if ( !v3 )
      JUMPOUT(0x418C16);
    while ( 1 )
    {
      j = *(_QWORD *)(v3 + 24);
      result = v3 + 16;
      if ( j != v3 + 16 )
        break;
      v3 = *(_QWORD *)(v3 + 8);
      if ( v4 == v3 )
        return result;
      if ( !v3 )
        BUG();
    }
    while ( 1 )
    {
      if ( v4 == v3 )
        return result;
      if ( !j )
        BUG();
      if ( (unsigned __int8)(*(_BYTE *)(j - 8) - 54) <= 1u )
      {
        v6 = v3;
        v21 = j - 24;
        if ( v4 != v3 )
        {
          v24 = v3;
          i = j;
          v8 = *(_BYTE *)(j - 8);
          v23 = j;
          v9 = v6;
          if ( (unsigned __int8)(v8 - 54) <= 1u )
            goto LABEL_27;
          while ( 1 )
          {
            for ( i = *(_QWORD *)(i + 8); i == v9 - 24 + 40; i = *(_QWORD *)(v9 + 24) )
            {
              v9 = *(_QWORD *)(v9 + 8);
              if ( v4 == v9 )
                goto LABEL_23;
              if ( !v9 )
                BUG();
            }
            if ( v4 == v9 )
              break;
            if ( !i )
              BUG();
            if ( (unsigned __int8)(*(_BYTE *)(i - 8) - 54) <= 1u )
            {
LABEL_27:
              v10 = *(void **)(a2 + 24);
              if ( *(_QWORD *)(a2 + 16) - (_QWORD)v10 <= 0xCu )
              {
                sub_16E7EE0(a2, "da analyze - ", 13);
              }
              else
              {
                qmemcpy(v10, "da analyze - ", 13);
                *(_QWORD *)(a2 + 24) += 13LL;
              }
              sub_13B1040(&v25, v22, v21, i - 24, 1);
              if ( v25 )
              {
                v12 = 1;
                sub_13A6390((__int64)v25, a2);
                v20 = v4;
                while ( (*(unsigned int (__fastcall **)(__m128i *))(v25->m128i_i64[0] + 40))(v25) >= v12 )
                {
                  if ( (*(unsigned __int8 (__fastcall **)(__m128i *, _QWORD))(v25->m128i_i64[0] + 80))(v25, v12) )
                  {
                    v13 = *(__m128i **)(a2 + 24);
                    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v13 <= 0x1Au )
                    {
                      v15 = sub_16E7EE0(a2, "da analyze - split level = ", 27);
                    }
                    else
                    {
                      si128 = _mm_load_si128((const __m128i *)&xmmword_4289960);
                      v15 = a2;
                      qmemcpy(&v13[1], "it level = ", 11);
                      *v13 = si128;
                      *(_QWORD *)(a2 + 24) += 27LL;
                    }
                    sub_16E7A90(v15, v12);
                    v16 = *(void **)(a2 + 24);
                    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v16 <= 0xDu )
                    {
                      v17 = sub_16E7EE0(a2, ", iteration = ", 14);
                    }
                    else
                    {
                      v17 = a2;
                      qmemcpy(v16, ", iteration = ", 14);
                      *(_QWORD *)(a2 + 24) += 14LL;
                    }
                    v18 = sub_13AEF00(v22, (__int64)v25, v12);
                    sub_1456620(v18, v17);
                    v19 = *(_WORD **)(a2 + 24);
                    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v19 <= 1u )
                    {
                      sub_16E7EE0(a2, "!\n", 2);
                    }
                    else
                    {
                      *v19 = 2593;
                      *(_QWORD *)(a2 + 24) += 2LL;
                    }
                  }
                  ++v12;
                }
                v4 = v20;
              }
              else
              {
                v11 = *(_QWORD *)(a2 + 24);
                if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v11) <= 5 )
                {
                  sub_16E7EE0(a2, "none!\n", 6);
                }
                else
                {
                  *(_DWORD *)v11 = 1701736302;
                  *(_WORD *)(v11 + 4) = 2593;
                  *(_QWORD *)(a2 + 24) += 6LL;
                }
              }
              if ( v25 )
                (*(void (__fastcall **)(__m128i *))(v25->m128i_i64[0] + 8))(v25);
            }
          }
LABEL_23:
          v3 = v24;
          j = v23;
        }
      }
      for ( j = *(_QWORD *)(j + 8); ; j = *(_QWORD *)(v3 + 24) )
      {
        result = v3 - 24 + 40;
        if ( j != result )
          break;
        v3 = *(_QWORD *)(v3 + 8);
        if ( v4 == v3 )
          return result;
        if ( !v3 )
          BUG();
      }
    }
  }
  return result;
}
