// Function: sub_847710
// Address: 0x847710
//
__int64 __fastcall sub_847710(__m128i *a1, __m128i *a2, unsigned int a3, FILE *a4)
{
  const __m128i *v5; // r12
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 result; // rax
  __int8 v12; // al
  __int64 v13; // r14
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // rsi
  __m128i *v17; // rax

  v5 = a2;
  if ( dword_4F077C4 != 2 || !(unsigned int)sub_8D3A70(a2) )
  {
    sub_8453D0(a1, a2, 0, 0, 1u, 1u, 0x20u, a3, a4);
    goto LABEL_3;
  }
  v12 = a2[8].m128i_i8[12];
  if ( v12 == 12 )
  {
    v13 = (__int64)a2;
    do
      v13 = *(_QWORD *)(v13 + 160);
    while ( *(_BYTE *)(v13 + 140) == 12 );
    if ( !dword_4D04964 )
      goto LABEL_10;
  }
  else
  {
    if ( !dword_4D04964 )
    {
      v13 = (__int64)a2;
LABEL_21:
      if ( (v12 & 0xFB) != 8 )
      {
LABEL_22:
        v5 = (const __m128i *)v13;
        goto LABEL_23;
      }
LABEL_10:
      if ( (sub_8D4C10(a2, dword_4F077C4 != 2) & 8) != 0 )
      {
        v16 = a1->m128i_i64[0];
        if ( a1->m128i_i64[0] != v13 && !(unsigned int)sub_8D97D0(v13, v16, 32, v14, v15) )
        {
          if ( (unsigned int)sub_6E5430() )
            sub_6858F0(0x201u, a4, a1->m128i_i64[0], (__int64)v5);
          goto LABEL_3;
        }
LABEL_24:
        sub_6FA3A0(a1, v16);
        goto LABEL_3;
      }
      goto LABEL_22;
    }
    if ( (v12 & 0xFB) != 8 )
    {
LABEL_23:
      v17 = sub_73C570(v5, 1);
      v16 = sub_72D600(v17);
      sub_842520(a1, v16, 0, 1u, 0x20u, a3);
      goto LABEL_24;
    }
    v13 = (__int64)a2;
  }
  if ( !(unsigned int)sub_8D4C10(a2, dword_4F077C4 != 2) )
  {
    v12 = a2[8].m128i_i8[12];
    goto LABEL_21;
  }
  if ( (unsigned int)sub_6E5430() )
    sub_685360(0x174u, a4, v13);
LABEL_3:
  result = dword_4A52070[0];
  if ( dword_4A52070[0] )
    return sub_6E6B60(a1, 0, v7, v8, v9, v10);
  return result;
}
