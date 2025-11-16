// Function: sub_255A360
// Address: 0x255a360
//
__int64 __fastcall sub_255A360(__int64 a1, __int64 a2)
{
  __m128i *v2; // r13
  __int64 result; // rax
  unsigned __int64 v4; // rax
  __int64 v5; // rdx
  int v6; // edx
  __int64 v7; // rax
  unsigned __int64 v8; // rdi
  __int64 *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  unsigned int v14; // [rsp+14h] [rbp-74h]
  char v15; // [rsp+1Fh] [rbp-69h] BYREF
  __int64 v16[3]; // [rsp+20h] [rbp-68h] BYREF
  _BYTE *v17; // [rsp+38h] [rbp-50h] BYREF
  __int64 v18; // [rsp+40h] [rbp-48h]
  _BYTE v19[64]; // [rsp+48h] [rbp-40h] BYREF

  v2 = (__m128i *)(a1 + 72);
  v15 = 0;
  if ( (unsigned __int8)sub_251C230(a2, (__int64 *)(a1 + 72), 0, 0, &v15, 0, 1) )
    return 1;
  v4 = sub_2527850(a2, v2, a1, &v15, 2u);
  v16[2] = v5;
  v16[1] = v4;
  if ( !(_BYTE)v5 )
    return 1;
  v6 = *(unsigned __int8 *)sub_250D070(v2);
  result = 1;
  if ( (unsigned int)(v6 - 12) > 1 )
  {
    v18 = 0x400000000LL;
    v7 = *(_QWORD *)(a1 + 72);
    v17 = v19;
    v8 = v7 & 0xFFFFFFFFFFFFFFFCLL;
    if ( (v7 & 3) == 3 )
      v8 = *(_QWORD *)(v8 + 24);
    v9 = (__int64 *)sub_BD5C60(v8);
    v16[0] = sub_A778C0(v9, 40, 0);
    sub_25594F0((__int64)&v17, v16, v10, v11, v12, v13);
    result = 1;
    if ( (_DWORD)v18 )
      result = sub_2516380(a2, v2->m128i_i64, (__int64)v17, (unsigned int)v18, 0);
    if ( v17 != v19 )
    {
      v14 = result;
      _libc_free((unsigned __int64)v17);
      return v14;
    }
  }
  return result;
}
