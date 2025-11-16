// Function: sub_38D62E0
// Address: 0x38d62e0
//
__int64 __fastcall sub_38D62E0(__int64 a1, __int64 a2, unsigned int a3, unsigned __int64 a4)
{
  __int64 v7; // rbx
  __int64 v8; // rax
  unsigned int *v9; // rsi
  __int64 (__fastcall *v10)(__int64); // rax
  __int64 v11; // rdx
  int v12; // r8d
  int v13; // r9d
  char v14; // al
  __int64 result; // rax
  int v16; // eax
  unsigned int v17; // edx
  __int64 v18; // rax
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // r12
  int v21; // r14d
  size_t v22; // r12
  __int64 v23; // rax
  __int64 v24; // rdi
  signed __int64 v25; // [rsp+18h] [rbp-78h] BYREF
  _QWORD v26[2]; // [rsp+20h] [rbp-70h] BYREF
  __int16 v27; // [rsp+30h] [rbp-60h]
  __m128i v28; // [rsp+40h] [rbp-50h] BYREF
  unsigned __int64 v29; // [rsp+50h] [rbp-40h]

  sub_38DDC00();
  v7 = sub_38D4BB0(a1, 0);
  sub_38D4150(a1, v7, *(unsigned int *)(v7 + 72));
  sub_39120A0(a1);
  v8 = *(unsigned int *)(a1 + 120);
  v9 = 0;
  if ( (_DWORD)v8 )
    v9 = *(unsigned int **)(*(_QWORD *)(a1 + 112) + 32 * v8 - 32);
  sub_38CB070((_QWORD *)a1, v9);
  v10 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 72LL);
  if ( v10 == sub_38D3BD0 )
  {
    LODWORD(v11) = 0;
    if ( *(_BYTE *)(a1 + 260) )
      v11 = *(_QWORD *)(a1 + 264);
  }
  else
  {
    LODWORD(v11) = v10(a1);
  }
  if ( sub_38CF2B0(a2, &v25, v11) )
  {
    v14 = 8 * a3;
    if ( 8 * a3 <= 0x3F
      && v25 > 0xFFFFFFFFFFFFFFFFLL >> (64 - v14)
      && ((v23 = 1LL << (v14 - 1), v25 < -v23) || v25 > v23 - 1) )
    {
      v24 = *(_QWORD *)(a1 + 8);
      v26[1] = &v25;
      v26[0] = "value evaluated as ";
      v27 = 3075;
      v28.m128i_i64[0] = (__int64)v26;
      LOWORD(v29) = 770;
      v28.m128i_i64[1] = (__int64)" is out of range.";
      return (__int64)sub_38BE3D0(v24, a4, (__int64)&v28);
    }
    else
    {
      return (*(__int64 (__fastcall **)(__int64, signed __int64, _QWORD))(*(_QWORD *)a1 + 424LL))(a1, v25, a3);
    }
  }
  else
  {
    v16 = 2;
    if ( a3 != 4 )
    {
      v16 = 3;
      if ( a3 <= 4 )
        v16 = a3 != 1;
    }
    v17 = *(_DWORD *)(v7 + 72);
    v28.m128i_i64[0] = a2;
    v28.m128i_i64[1] = __PAIR64__(v16, v17);
    v29 = a4;
    v18 = *(unsigned int *)(v7 + 120);
    if ( (unsigned int)v18 >= *(_DWORD *)(v7 + 124) )
    {
      sub_16CD150(v7 + 112, (const void *)(v7 + 128), 0, 24, v12, v13);
      v18 = *(unsigned int *)(v7 + 120);
    }
    result = *(_QWORD *)(v7 + 112) + 24 * v18;
    *(__m128i *)result = _mm_loadu_si128(&v28);
    *(_QWORD *)(result + 16) = v29;
    v19 = *(unsigned int *)(v7 + 72);
    ++*(_DWORD *)(v7 + 120);
    v20 = v19 + a3;
    v21 = v19;
    if ( v20 > v19 )
    {
      result = v19;
      if ( v20 > *(unsigned int *)(v7 + 76) )
      {
        sub_16CD150(v7 + 64, (const void *)(v7 + 80), v20, 1, v12, v13);
        result = *(unsigned int *)(v7 + 72);
        v19 = result;
      }
      v22 = v20 - result;
      if ( v22 )
        result = (__int64)memset((void *)(*(_QWORD *)(v7 + 64) + v19), 0, v22);
      *(_DWORD *)(v7 + 72) = v21 + a3;
    }
  }
  return result;
}
