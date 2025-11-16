// Function: sub_16F79B0
// Address: 0x16f79b0
//
void __fastcall sub_16F79B0(__int64 a1, __int64 a2, int a3, char a4, int a5, int a6)
{
  __int64 v6; // rbp
  __int64 v7; // rax
  __int64 v8; // rax
  __m128i *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // [rsp-38h] [rbp-38h] BYREF
  int v12; // [rsp-30h] [rbp-30h]
  _BYTE v13[12]; // [rsp-2Ch] [rbp-2Ch]
  __int64 v14; // [rsp-8h] [rbp-8h]

  if ( *(_BYTE *)(a1 + 73) )
  {
    v14 = v6;
    v7 = *(_QWORD *)(a1 + 64);
    v11 = a2;
    v12 = a3;
    *(_QWORD *)v13 = v7;
    v8 = *(unsigned int *)(a1 + 240);
    v13[8] = a4;
    if ( (unsigned int)v8 >= *(_DWORD *)(a1 + 244) )
    {
      sub_16CD150(a1 + 232, (const void *)(a1 + 248), 0, 24, a5, a6);
      v8 = *(unsigned int *)(a1 + 240);
    }
    v9 = (__m128i *)(*(_QWORD *)(a1 + 232) + 24 * v8);
    v10 = *(_QWORD *)&v13[4];
    *v9 = _mm_loadu_si128((const __m128i *)&v11);
    v9[1].m128i_i64[0] = v10;
    ++*(_DWORD *)(a1 + 240);
  }
}
