// Function: sub_A690C0
// Address: 0xa690c0
//
__int64 __fastcall sub_A690C0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  const __m128i *v5; // rbx
  __int64 v6; // rsi
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v11; // [rsp-10h] [rbp-500h]
  _BYTE v13[112]; // [rsp+10h] [rbp-4E0h] BYREF
  _BYTE v14[400]; // [rsp+80h] [rbp-470h] BYREF
  __int64 v15[92]; // [rsp+210h] [rbp-2E0h] BYREF

  v5 = (const __m128i *)v14;
  sub_A54BD0((__int64)v13, a2);
  sub_A55A10((__int64)v14, 0, 0);
  if ( sub_A56340(a3, 0) )
    v5 = sub_A56340(a3, 0);
  if ( !*(_QWORD *)(a1 + 16) )
    goto LABEL_11;
  if ( ((__int64 (*)(void))sub_B14180)() )
  {
    v6 = *(_QWORD *)(sub_B14180(*(_QWORD *)(a1 + 16)) + 72);
    if ( v6 )
      sub_A564B0(a3, v6);
  }
  v7 = *(_QWORD *)(a1 + 16);
  if ( v7 && sub_B14170(*(_QWORD *)(a1 + 16)) && (v8 = *(_QWORD *)(sub_B14170(v7) + 72)) != 0 )
    v9 = *(_QWORD *)(v8 + 40);
  else
LABEL_11:
    v9 = 0;
  sub_A685A0((__int64)v15, (__int64)v13, (__int64)v5, v9, 0, a4, 0);
  sub_A5C700(v15, a1);
  sub_A555E0((__int64)v15);
  sub_A552A0((__int64)v14, a1);
  sub_A54D10((__int64)v13, a1);
  return v11;
}
