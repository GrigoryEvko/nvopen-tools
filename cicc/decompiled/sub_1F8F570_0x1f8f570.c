// Function: sub_1F8F570
// Address: 0x1f8f570
//
__int64 *__fastcall sub_1F8F570(_QWORD *a1, __int64 a2, double a3, __m128i a4, __m128i a5)
{
  unsigned int v6; // r13d
  __int64 *v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rsi
  __int64 v10; // rbx
  __int64 (*v11)(); // rax

  v6 = *(unsigned __int16 *)(a2 + 80);
  if ( ((*(_BYTE *)(*(_QWORD *)*a1 + 792LL) & 2) != 0 || (*(_BYTE *)(a2 + 81) & 4) != 0)
    && ((v7 = *(__int64 **)(a2 + 32),
         v8 = a1[1],
         v9 = *v7,
         v10 = v7[1],
         v11 = *(__int64 (**)())(*(_QWORD *)v8 + 96LL),
         v11 == sub_1F3C9B0)
     || !((unsigned __int8 (__fastcall *)(__int64, __int64, __int64))v11)(v8, v9, v10)) )
  {
    return sub_1F8E5A0(a1, v9, v10, v6, 0, a3, a4, a5);
  }
  else
  {
    return 0;
  }
}
