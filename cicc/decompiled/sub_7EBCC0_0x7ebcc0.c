// Function: sub_7EBCC0
// Address: 0x7ebcc0
//
_QWORD *__fastcall sub_7EBCC0(_QWORD **a1, __int64 *a2, unsigned int a3, __int64 a4, const __m128i **a5, __int64 a6)
{
  int v10; // eax
  const __m128i *v11; // rdi
  _QWORD *v12; // r13
  _QWORD *v13; // r14
  __int64 v14; // rsi
  _BYTE *v15; // rax
  _BYTE *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  _QWORD *result; // rax
  const __m128i *v22; // rax

  *a5 = 0;
  v10 = sub_731920(*a2, a3, 0, a4, (__int64)a5, a6);
  v11 = (const __m128i *)*a2;
  if ( !v10 )
  {
    *a5 = v11;
    v22 = (const __m128i *)sub_7E88C0(v11);
    *a2 = (__int64)v22;
    v11 = v22;
  }
  v12 = sub_7E8090(v11, a3);
  v13 = sub_7EBC30((__int64)a1, (_QWORD *)*a2);
  v14 = sub_72D2E0(*a1);
  v15 = sub_73E130(v13, v14);
  v16 = sub_73DCD0(v15);
  result = sub_731370((__int64)v16, v14, v17, v18, v19, v20);
  *a2 = (__int64)v12;
  return result;
}
