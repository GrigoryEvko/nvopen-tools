// Function: sub_7F5690
// Address: 0x7f5690
//
_QWORD *__fastcall sub_7F5690(__int64 a1, __int64 a2, __int64 a3)
{
  const __m128i *v5; // rax
  __int64 v6; // rdi
  __int64 v7; // r12
  _QWORD *v8; // rax
  __int64 v9; // r15
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // rbx
  const __m128i *v15; // rdi
  _QWORD *v16; // r12
  const __m128i *v18; // [rsp+8h] [rbp-38h] BYREF

  v5 = (const __m128i *)sub_724DC0();
  v6 = *(_QWORD *)(a1 + 120);
  v7 = (__int64)v5;
  v18 = v5;
  v8 = (_QWORD *)sub_8D4050(v6);
  v9 = sub_72D2E0(v8);
  sub_72D510(a1, v7, 0);
  sub_70FEE0(v7, v9, v10, v11, v12);
  sub_72A420((__int64 *)a1);
  *(_BYTE *)(a1 + 88) |= 4u;
  v13 = sub_7E3470(a2, a3);
  v14 = v13;
  if ( a3 )
    v14 = *(_QWORD *)(a3 + 128) + v13;
  v15 = v18;
  *(_QWORD *)(v7 + 192) = sub_7E1340() * v14;
  v16 = sub_73A720(v15, a3);
  sub_724E30((__int64)&v18);
  return v16;
}
