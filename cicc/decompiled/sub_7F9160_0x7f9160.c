// Function: sub_7F9160
// Address: 0x7f9160
//
__int64 *__fastcall sub_7F9160(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 v4; // rax
  const __m128i *v5; // rdi
  __int64 *v6; // r12
  __m128i *v7; // rax
  __int64 v8; // rsi
  _QWORD *v9; // rax
  __int64 v10; // rsi
  _QWORD *v12; // r12
  _QWORD *v13; // rax
  __int64 v14; // r12
  __m128i *v15[3]; // [rsp+8h] [rbp-18h] BYREF

  v2 = *(_QWORD *)(a1 + 8);
  if ( !v2 || (*(_BYTE *)(v2 + 173) & 4) == 0 || *(_BYTE *)(a1 + 16) )
  {
    if ( *(_BYTE *)(a1 + 17) )
    {
      v3 = *(_QWORD *)(a1 + 40);
      v4 = v3;
LABEL_5:
      if ( v4 || (v5 = *(const __m128i **)(a1 + 56)) == 0 )
      {
        v15[0] = (__m128i *)sub_724DC0();
        sub_7E2DB0(v15[0], v3, byte_4F06A51[0], 0, 0);
        v6 = sub_73A720(v15[0], v3);
        sub_724E30((__int64)v15);
      }
      else
      {
        v6 = sub_7E8090(v5, 1u);
      }
      if ( !v6 )
        return 0;
      goto LABEL_9;
    }
LABEL_18:
    v14 = sub_7F9140(a1);
    if ( !(unsigned int)sub_8D3410(v14) )
      return 0;
    v3 = sub_8D4490(v14);
    v4 = *(_QWORD *)(a1 + 40);
    goto LABEL_5;
  }
  if ( *(_QWORD *)(a1 + 32) && !*(_BYTE *)(a1 + 17) )
    goto LABEL_18;
  v12 = sub_73E830(*(_QWORD *)(v2 + 248));
  v13 = sub_72BA30(byte_4F06A51[0]);
  v6 = (__int64 *)sub_73E130(v12, (__int64)v13);
  if ( !v6 )
    return 0;
LABEL_9:
  if ( *(__int64 *)(a1 + 64) > 0 )
  {
    v7 = (__m128i *)sub_724DC0();
    v8 = *(_QWORD *)(a1 + 64);
    v15[0] = v7;
    sub_7E2DB0(v7, v8, byte_4F06A51[0], 0, 0);
    v9 = sub_73A720(v15[0], v8);
    v10 = *v6;
    v6[2] = (__int64)v9;
    v6 = (__int64 *)sub_73DBF0(0x28u, v10, (__int64)v6);
    sub_724E30((__int64)v15);
  }
  return v6;
}
