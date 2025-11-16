// Function: sub_7E25D0
// Address: 0x7e25d0
//
void __fastcall sub_7E25D0(__int64 a1, int *a2)
{
  __int64 *v2; // rbx
  int v3; // r15d
  const __m128i *v4; // r13
  __int64 *v5; // rax
  __int64 v6; // r14
  __int64 v7; // r15
  char v8; // r12
  __int8 v9; // bl
  __int64 v10; // [rsp+8h] [rbp-38h]

  v2 = (__int64 *)a1;
  v3 = *a2;
  if ( *a2 == 5 )
  {
    *((_QWORD *)a2 + 1) = a1;
    *a2 = 4;
  }
  else
  {
    v4 = (const __m128i *)*((_QWORD *)a2 + 1);
    v5 = (__int64 *)sub_730FF0(v4);
    v6 = (__int64)v5;
    if ( v3 == 3 )
    {
      *((_QWORD *)a2 + 1) = v5;
      v6 = a1;
      v2 = v5;
    }
    *(_QWORD *)(v6 + 16) = v2;
    v7 = *v2;
    v2[2] = 0;
    v8 = *((_BYTE *)v2 + 25);
    v9 = v4[1].m128i_i8[9];
    v10 = v4[1].m128i_i64[0];
    sub_7266C0((__int64)v4, 1);
    v4[1].m128i_i64[0] = v10;
    v4[1].m128i_i8[9] = v4[1].m128i_i8[9] & 0xFB | (4 * ((v9 & 4) != 0));
    sub_73D8E0((__int64)v4, 0x5Bu, v7, v8 & 1, v6);
  }
}
