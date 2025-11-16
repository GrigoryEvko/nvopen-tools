// Function: sub_B815A0
// Address: 0xb815a0
//
__int64 __fastcall sub_B815A0(__int64 a1, int a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __m128i *v4; // rdx
  __m128i si128; // xmm0
  __int64 result; // rax
  unsigned int v7; // r13d
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // rsi
  unsigned int v11; // ecx
  __int64 *v12; // rdx
  __int64 v13; // r8
  __int64 v14; // rax
  __int64 v15; // r15
  __int64 (__fastcall *v16)(__int64, unsigned int); // rax
  unsigned int v17; // r14d
  __int64 v18; // rax
  int v19; // edx
  int v20; // r9d
  unsigned int i; // [rsp+8h] [rbp-38h]
  unsigned int v22; // [rsp+Ch] [rbp-34h]

  v2 = sub_C5F790();
  v3 = sub_CB69B0(v2, (unsigned int)(2 * a2));
  v4 = *(__m128i **)(v3 + 32);
  if ( *(_QWORD *)(v3 + 24) - (_QWORD)v4 <= 0x12u )
  {
    sub_CB6200(v3, "ModulePass Manager\n", 19);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F552A0);
    v4[1].m128i_i8[2] = 10;
    v4[1].m128i_i16[0] = 29285;
    *v4 = si128;
    *(_QWORD *)(v3 + 32) += 19LL;
  }
  result = (unsigned int)(a2 + 1);
  v22 = 0;
  v7 = a2 + 2;
  for ( i = a2 + 1; *(_DWORD *)(a1 + 200) > v22; result = v22 )
  {
    v8 = *(_QWORD *)(*(_QWORD *)(a1 + 192) + 8LL * v22);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v8 + 136LL))(v8, i);
    v9 = *(unsigned int *)(a1 + 592);
    v10 = *(_QWORD *)(a1 + 576);
    if ( (_DWORD)v9 )
    {
      v11 = (v9 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v12 = (__int64 *)(v10 + 16LL * v11);
      v13 = *v12;
      if ( v8 == *v12 )
      {
LABEL_6:
        if ( v12 != (__int64 *)(v10 + 16 * v9) )
        {
          v14 = *(_QWORD *)(a1 + 600) + 16LL * *((unsigned int *)v12 + 2);
          if ( v14 != *(_QWORD *)(a1 + 600) + 16LL * *(unsigned int *)(a1 + 608) )
          {
            v15 = *(_QWORD *)(v14 + 8);
            v16 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v15 + 136LL);
            if ( v16 == sub_B7E5C0 )
            {
              if ( *(_DWORD *)(v15 + 608) )
              {
                v17 = 0;
                do
                {
                  v18 = *(_QWORD *)(*(_QWORD *)(v15 + 600) + 8LL * v17);
                  if ( !v18 )
                    BUG();
                  ++v17;
                  (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)(v18 - 176) + 136LL))(v18 - 176, v7);
                }
                while ( *(_DWORD *)(v15 + 608) > v17 );
              }
            }
            else
            {
              v16(v15, v7);
            }
          }
        }
      }
      else
      {
        v19 = 1;
        while ( v13 != -4096 )
        {
          v20 = v19 + 1;
          v11 = (v9 - 1) & (v19 + v11);
          v12 = (__int64 *)(v10 + 16LL * v11);
          v13 = *v12;
          if ( v8 == *v12 )
            goto LABEL_6;
          v19 = v20;
        }
      }
    }
    sub_B81320(a1 + 176, v8, i);
    ++v22;
  }
  return result;
}
