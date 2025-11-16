// Function: sub_8283A0
// Address: 0x8283a0
//
__int64 __fastcall sub_8283A0(__int64 a1, __int64 a2, int a3, int a4)
{
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r14
  __int64 *v13; // r13
  __int64 *v15; // r9
  _BYTE *v16; // rsi
  _BYTE *v17; // rax
  __int64 v18; // [rsp+10h] [rbp-30h] BYREF
  __int64 v19[5]; // [rsp+18h] [rbp-28h] BYREF

  v6 = a2;
  if ( !a2 )
    v6 = *(_QWORD *)a1;
  if ( (unsigned int)sub_8D3410(v6) )
  {
    v7 = sub_6ECAE0(v6, a3, a4, 0, 7u, (__int64 *)(a1 + 68), &v18);
  }
  else
  {
    sub_6FA3A0((__m128i *)a1, a2);
    v15 = (__int64 *)(a1 + 68);
    if ( *(_BYTE *)(a1 + 16) == 2 && (*(_BYTE *)(a1 + 315) & 2) != 0 )
    {
      v13 = (__int64 *)sub_6ECAE0(v6, a3, a4, 0, 2u, v15, &v18);
      v16 = sub_724DC0();
      v19[0] = (__int64)v16;
      sub_6F4950((__m128i *)a1, (__int64)v16);
      v17 = (_BYTE *)sub_724E50(v19, v16);
      sub_72F900(v18, v17);
      goto LABEL_6;
    }
    v7 = sub_6ECAE0(v6, a3, a4, 0, 3u, v15, &v18);
  }
  v12 = v18;
  v13 = (__int64 *)v7;
  *(_QWORD *)(v12 + 56) = sub_6F6F40((const __m128i *)a1, 0, v8, v9, v10, v11);
LABEL_6:
  sub_6E7170(v13, a1);
  return sub_6E26D0(2, a1);
}
