// Function: sub_C299E0
// Address: 0xc299e0
//
__int64 __fastcall sub_C299E0(_QWORD *a1)
{
  __int64 result; // rax
  unsigned int v3; // r13d
  _QWORD *v4; // rsi
  int v5; // ebx
  int v6; // r14d
  unsigned int v7; // eax
  __int64 v8; // rax
  __m128i *v9; // rdx
  __int64 v10; // rdi
  __m128i si128; // xmm0
  __int64 v12; // rdi
  _BYTE *v13; // rax
  _QWORD v14[2]; // [rsp+10h] [rbp-90h] BYREF
  _BYTE v15[128]; // [rsp+20h] [rbp-80h] BYREF

  result = sub_C219E0((__int64)a1, 2885681152LL);
  if ( !(_DWORD)result )
  {
    v3 = 0;
    if ( (unsigned __int64)(a1[29] + 4LL) > a1[27] )
    {
      v8 = sub_CB72A0(a1, 2885681152LL);
      v9 = *(__m128i **)(v8 + 32);
      v10 = v8;
      if ( *(_QWORD *)(v8 + 24) - (_QWORD)v9 <= 0x20u )
      {
        v10 = sub_CB6200(v8, "unexpected end of memory buffer: ", 33);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_3F64EF0);
        v9[2].m128i_i8[0] = 32;
        *v9 = si128;
        v9[1] = _mm_load_si128((const __m128i *)&xmmword_3F64F00);
        *(_QWORD *)(v8 + 32) += 33LL;
      }
      v12 = sub_CB59D0(v10, a1[29]);
      v13 = *(_BYTE **)(v12 + 32);
      if ( *(_BYTE **)(v12 + 24) == v13 )
      {
        sub_CB6200(v12, "\n", 1);
      }
      else
      {
        *v13 = 10;
        ++*(_QWORD *)(v12 + 32);
      }
      v3 = 4;
      sub_C1AFD0();
    }
    else
    {
      v4 = a1 + 29;
      v5 = sub_C5F610(a1 + 26, a1 + 29, a1 + 30);
      v14[0] = v15;
      v14[1] = 0xA00000000LL;
      if ( v5 )
      {
        v6 = 0;
        while ( 1 )
        {
          v4 = v14;
          v7 = sub_C29010(a1, (__int64)v14, 1u, 0);
          if ( v7 )
            break;
          if ( ++v6 == v5 )
            goto LABEL_11;
        }
        v3 = v7;
      }
      else
      {
LABEL_11:
        sub_C21C30((__int64)a1);
        sub_C1AFD0();
      }
      if ( (_BYTE *)v14[0] != v15 )
        _libc_free(v14[0], v4);
    }
    return v3;
  }
  return result;
}
