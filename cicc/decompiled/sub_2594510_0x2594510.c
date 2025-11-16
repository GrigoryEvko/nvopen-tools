// Function: sub_2594510
// Address: 0x2594510
//
__int64 __fastcall sub_2594510(__int64 a1, _QWORD *a2)
{
  unsigned int v2; // r15d
  __m128i *v3; // r14
  unsigned __int64 v7; // rdi
  __int64 *v8; // r15
  char v9; // al
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  char v16; // [rsp+17h] [rbp-69h] BYREF
  __int64 v17; // [rsp+18h] [rbp-68h] BYREF
  _BYTE *v18; // [rsp+20h] [rbp-60h] BYREF
  __int64 v19; // [rsp+28h] [rbp-58h]
  _BYTE v20[80]; // [rsp+30h] [rbp-50h] BYREF

  v2 = 1;
  v3 = (__m128i *)(a1 + 72);
  if ( (unsigned int)*(unsigned __int8 *)sub_250D070((_QWORD *)(a1 + 72)) - 12 > 1 )
  {
    v18 = v20;
    v19 = 0x400000000LL;
    v7 = *(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFFCLL;
    if ( (*(_QWORD *)(a1 + 72) & 3LL) == 3 )
      v7 = *(_QWORD *)(v7 + 24);
    v8 = (__int64 *)sub_BD5C60(v7);
    v9 = sub_258F340(a2, a1, v3, 2, &v16, 0, 0);
    v10 = *(unsigned int *)(a1 + 108);
    if ( v9 )
      v11 = sub_A77A80(v8, v10);
    else
      v11 = sub_A77A90(v8, v10);
    v17 = v11;
    v2 = 1;
    sub_25594F0((__int64)&v18, &v17, v12, v13, v14, v15);
    if ( (_DWORD)v19 )
      v2 = sub_2516380((__int64)a2, v3->m128i_i64, (__int64)v18, (unsigned int)v19, 0);
    if ( v18 != v20 )
      _libc_free((unsigned __int64)v18);
  }
  if ( (unsigned __int8)sub_258F340(a2, a1, v3, 2, &v17, 0, 0) )
  {
    LODWORD(v18) = 91;
    if ( (unsigned __int8)sub_2516400((__int64)a2, v3, (__int64)&v18, 1, 0, 0) )
    {
      LODWORD(v18) = 91;
      v2 = 0;
      sub_2515E10((__int64)a2, v3->m128i_i64, (__int64)&v18, 1);
    }
  }
  return v2;
}
