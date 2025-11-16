// Function: sub_379CED0
// Address: 0x379ced0
//
unsigned __int8 *__fastcall sub_379CED0(__int64 *a1, __int64 a2, __m128i a3)
{
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 (__fastcall *v8)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v9; // rax
  unsigned __int16 v10; // si
  __int64 v11; // r8
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rsi
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  int v18; // eax
  _QWORD *v19; // r13
  unsigned int v20; // edx
  __int64 v21; // rdx
  __int64 v22; // rdx
  int v23; // [rsp+0h] [rbp-40h] BYREF
  __int64 v24; // [rsp+8h] [rbp-38h]
  __int64 v25; // [rsp+10h] [rbp-30h]

  v4 = *(_QWORD *)(a2 + 40);
  v5 = *(_QWORD *)(*(_QWORD *)(v4 + 40) + 48LL) + 16LL * *(unsigned int *)(v4 + 48);
  v6 = *(_QWORD *)(*(_QWORD *)v4 + 48LL) + 16LL * *(unsigned int *)(v4 + 8);
  if ( *(_WORD *)v6 == *(_WORD *)v5 && (*(_QWORD *)(v6 + 8) == *(_QWORD *)(v5 + 8) || *(_WORD *)v6) )
    return sub_379C350(a1, a2);
  v8 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v9 = *(__int16 **)(a2 + 48);
  v10 = *v9;
  v11 = *((_QWORD *)v9 + 1);
  v12 = a1[1];
  if ( v8 == sub_2D56A50 )
  {
    v13 = v10;
    v14 = *a1;
    sub_2FE6CC0((__int64)&v23, *a1, *(_QWORD *)(v12 + 64), v13, v11);
    LOWORD(v18) = v24;
    LOWORD(v23) = v24;
    v24 = v25;
  }
  else
  {
    v21 = v10;
    v14 = *(_QWORD *)(v12 + 64);
    v18 = v8(*a1, v14, v21, v11);
    v23 = v18;
    v24 = v22;
  }
  v19 = (_QWORD *)a1[1];
  if ( !(_WORD)v18 )
  {
    if ( !sub_3007100((__int64)&v23) )
      goto LABEL_11;
    goto LABEL_14;
  }
  if ( (unsigned __int16)(v18 - 176) <= 0x34u )
  {
LABEL_14:
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( (_WORD)v23 )
    {
      if ( (unsigned __int16)(v23 - 176) <= 0x34u )
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
      goto LABEL_9;
    }
LABEL_11:
    v20 = sub_3007130((__int64)&v23, v14);
    return sub_3412A00(v19, a2, v20, v15, v16, v17, a3);
  }
LABEL_9:
  v20 = word_4456340[(unsigned __int16)v23 - 1];
  return sub_3412A00(v19, a2, v20, v15, v16, v17, a3);
}
