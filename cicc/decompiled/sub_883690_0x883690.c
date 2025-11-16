// Function: sub_883690
// Address: 0x883690
//
__int64 __fastcall sub_883690(const __m128i *a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // rbx
  unsigned __int8 *v9; // rdi
  _QWORD *v10; // r14
  __int64 v11; // rax
  __int64 v12; // rbx
  __m128i *v13; // r12
  __int64 result; // rax
  __int64 v15; // rax
  _DWORD v16[12]; // [rsp+0h] [rbp-30h] BYREF

  v7 = (int)dword_4F04C44;
  if ( dword_4F04C44 == -1 )
    v7 = unk_4F04C48;
  v8 = *(_QWORD *)(qword_4F04C68[0] + 776 * v7 + 408);
  v9 = *(unsigned __int8 **)(v8 + 88);
  if ( !v9 )
  {
    v15 = sub_881A70(0, 1u, 34, 35, a5, a6);
    *(_QWORD *)(v8 + 88) = v15;
    v9 = (unsigned __int8 *)v15;
  }
  v16[0] = a2;
  v10 = (_QWORD *)sub_881B20(v9, (__int64)v16, 1);
  if ( *v10 )
    sub_721090();
  v11 = sub_823970(16);
  *(_DWORD *)v11 = a2;
  v12 = v11;
  *(_QWORD *)(v11 + 8) = 0;
  v13 = (__m128i *)sub_823970(24);
  sub_878D60(v13);
  *v13 = _mm_loadu_si128(a1);
  result = a1[1].m128i_i64[0];
  v13[1].m128i_i64[0] = result;
  *(_QWORD *)(v12 + 8) = v13;
  *v10 = v12;
  return result;
}
