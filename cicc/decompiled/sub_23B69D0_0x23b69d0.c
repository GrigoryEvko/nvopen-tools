// Function: sub_23B69D0
// Address: 0x23b69d0
//
void __fastcall sub_23B69D0(__int64 a1, __int64 *a2, __m128i si128)
{
  __int64 v4; // r12
  _QWORD *(__fastcall *v5)(__int64 *, __int64); // rax
  _QWORD *v6; // rax
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rbx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 (__fastcall **v14)(); // r12
  __int64 v15; // rdi
  __m128 *v16; // rdx
  __int64 v17[5]; // [rsp+18h] [rbp-28h] BYREF

  v4 = *a2;
  if ( *a2 )
  {
    v5 = *(_QWORD *(__fastcall **)(__int64 *, __int64))(*(_QWORD *)v4 + 16LL);
    if ( v5 == sub_23AEE80 )
    {
      v6 = (_QWORD *)sub_22077B0(0x68u);
      v10 = (__int64)v6;
      if ( v6 )
      {
        *v6 = &unk_4A16218;
        sub_C8CD80((__int64)(v6 + 1), (__int64)(v6 + 5), v4 + 8, v7, v8, v9);
        sub_C8CD80(v10 + 56, v10 + 88, v4 + 56, v11, v12, v13);
      }
      v17[0] = v10;
    }
    else
    {
      v5(v17, *a2);
    }
  }
  else
  {
    v17[0] = 0;
  }
  v14 = (__int64 (__fastcall **)())sub_23B66B0(v17, 1);
  if ( v17[0] )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v17[0] + 8LL))(v17[0]);
  v15 = *(_QWORD *)(a1 + 40);
  v16 = *(__m128 **)(v15 + 32);
  if ( *(_QWORD *)(v15 + 24) - (_QWORD)v16 <= 0x18u )
  {
    sub_CB6200(v15, "*** IR Dump At Start ***\n", 0x19u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_437F160);
    v16[1].m128_i8[8] = 10;
    v16[1].m128_u64[0] = 0x2A2A2A2074726174LL;
    *v16 = (__m128)si128;
    *(_QWORD *)(v15 + 32) += 25LL;
  }
  sub_A69980(v14, *(_QWORD *)(a1 + 40), 0, 0, 0, si128);
}
