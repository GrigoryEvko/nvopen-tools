// Function: sub_A69280
// Address: 0xa69280
//
__int64 __fastcall sub_A69280(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  const __m128i *v5; // rbx
  __int64 v6; // rsi
  __int64 v7; // r13
  char v8; // r9
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v14; // [rsp-10h] [rbp-500h]
  _BYTE v16[112]; // [rsp+10h] [rbp-4E0h] BYREF
  _BYTE v17[400]; // [rsp+80h] [rbp-470h] BYREF
  __int64 v18[92]; // [rsp+210h] [rbp-2E0h] BYREF

  v5 = (const __m128i *)v17;
  sub_A54BD0((__int64)v16, a2);
  sub_A55A10((__int64)v17, 0, 0);
  if ( sub_A56340(a3, 0) )
    v5 = sub_A56340(a3, 0);
  if ( sub_B14180(*(_QWORD *)(a1 + 16)) )
  {
    v6 = *(_QWORD *)(sub_B14180(*(_QWORD *)(a1 + 16)) + 72);
    if ( v6 )
      sub_A564B0(a3, v6);
  }
  v7 = *(_QWORD *)(a1 + 16);
  v8 = a4;
  if ( v7
    && (v9 = sub_B14170(*(_QWORD *)(a1 + 16)), v8 = a4, v9)
    && (v10 = sub_B14170(v7), v8 = a4, (v11 = *(_QWORD *)(v10 + 72)) != 0) )
  {
    v12 = *(_QWORD *)(v11 + 40);
  }
  else
  {
    v12 = 0;
  }
  sub_A685A0((__int64)v18, (__int64)v16, (__int64)v5, v12, 0, v8, 0);
  sub_A5C590(v18, a1);
  sub_A555E0((__int64)v18);
  sub_A552A0((__int64)v17, a1);
  sub_A54D10((__int64)v16, a1);
  return v14;
}
