// Function: sub_31DA6E0
// Address: 0x31da6e0
//
__int64 (*__fastcall sub_31DA6E0(__int64 a1, __int64 a2))()
{
  __int64 v3; // rdi
  __int64 (*result)(); // rax
  __int64 (*v5)(); // r14
  __int64 v6; // r12
  __int64 v7; // r13
  __int64 v8; // rcx
  unsigned __int64 v9; // rax
  int v10; // edx
  __int64 v11; // rsi
  __int64 v12; // rdi
  __int64 v13; // r8
  __int64 v14; // r9
  int v15; // edi
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // rcx
  const __m128i *v18; // r15
  __m128i *v19; // rax
  __m128i v20; // xmm1
  unsigned int v21; // r15d
  __int64 *v22; // rax
  __int64 v23; // r13
  void (__fastcall *v24)(__int64, __int64); // r14
  __int64 v25; // rax
  __int64 v26; // rax
  unsigned __int8 v27; // dl
  __int64 v28; // rdi
  __int64 v29; // r9
  char *v30; // r15
  int v31; // [rsp+8h] [rbp-58h]
  __int64 v32; // [rsp+10h] [rbp-50h] BYREF
  int v33; // [rsp+18h] [rbp-48h]
  __int64 v34; // [rsp+20h] [rbp-40h]
  int v35; // [rsp+28h] [rbp-38h]

  v3 = sub_31DA6B0(a1);
  result = *(__int64 (**)())(*(_QWORD *)v3 + 216LL);
  if ( result != sub_302E490 )
  {
    result = (__int64 (*)())((__int64 (__fastcall *)(__int64))result)(v3);
    v5 = result;
    if ( result )
    {
      result = (__int64 (*)())sub_BA8DC0(a2, (__int64)"llvm.commandline", 16);
      v6 = (__int64)result;
      if ( result )
      {
        result = (__int64 (*)())sub_B91A00((__int64)result);
        if ( (_DWORD)result )
        {
          v7 = *(_QWORD *)(a1 + 224);
          v8 = *(unsigned int *)(v7 + 128);
          v9 = *(_QWORD *)(v7 + 120);
          v10 = *(_DWORD *)(v7 + 128);
          v11 = 32 * v8;
          if ( (_DWORD)v8 )
          {
            v12 = v9 + v11 - 32;
            v13 = *(_QWORD *)(v12 + 16);
            v10 = *(_DWORD *)(v12 + 24);
            v14 = *(_QWORD *)v12;
            v15 = *(_DWORD *)(v12 + 8);
          }
          else
          {
            v15 = 0;
            v14 = 0;
            v13 = 0;
          }
          v35 = v10;
          v16 = v8 + 1;
          v17 = *(unsigned int *)(v7 + 132);
          v18 = (const __m128i *)&v32;
          v32 = v14;
          v33 = v15;
          v34 = v13;
          if ( v16 > v17 )
          {
            v28 = v7 + 120;
            v29 = v7 + 136;
            if ( v9 > (unsigned __int64)&v32 || (unsigned __int64)&v32 >= v9 + v11 )
            {
              sub_C8D5F0(v28, (const void *)(v7 + 136), v16, 0x20u, (__int64)&v32, v29);
              v9 = *(_QWORD *)(v7 + 120);
              v11 = 32LL * *(unsigned int *)(v7 + 128);
            }
            else
            {
              v30 = (char *)&v32 - v9;
              sub_C8D5F0(v28, (const void *)(v7 + 136), v16, 0x20u, (__int64)&v32, v29);
              v9 = *(_QWORD *)(v7 + 120);
              v18 = (const __m128i *)&v30[v9];
              v11 = 32LL * *(unsigned int *)(v7 + 128);
            }
          }
          v19 = (__m128i *)(v11 + v9);
          *v19 = _mm_loadu_si128(v18);
          v20 = _mm_loadu_si128(v18 + 1);
          v21 = 0;
          v19[1] = v20;
          ++*(_DWORD *)(v7 + 128);
          (*(void (__fastcall **)(_QWORD, __int64 (*)(), _QWORD))(**(_QWORD **)(a1 + 224) + 176LL))(
            *(_QWORD *)(a1 + 224),
            v5,
            0);
          sub_E99300(*(_QWORD ***)(a1 + 224), 1);
          v31 = sub_B91A00(v6);
          if ( v31 )
          {
            do
            {
              v26 = sub_B91A10(v6, v21);
              v27 = *(_BYTE *)(v26 - 16);
              if ( (v27 & 2) != 0 )
                v22 = *(__int64 **)(v26 - 32);
              else
                v22 = (__int64 *)(-16 - 8LL * ((v27 >> 2) & 0xF) + v26);
              v23 = *(_QWORD *)(a1 + 224);
              ++v21;
              v24 = *(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v23 + 512LL);
              v25 = sub_B91420(*v22);
              v24(v23, v25);
              sub_E99300(*(_QWORD ***)(a1 + 224), 1);
            }
            while ( v31 != v21 );
          }
          return (__int64 (*)())(*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 224) + 168LL))(*(_QWORD *)(a1 + 224));
        }
      }
    }
  }
  return result;
}
