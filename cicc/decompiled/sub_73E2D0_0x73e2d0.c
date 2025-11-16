// Function: sub_73E2D0
// Address: 0x73e2d0
//
__int64 __fastcall sub_73E2D0(__int64 a1, __int64 a2)
{
  __int64 *v3; // r14
  __int64 v4; // r13
  __m128i *v5; // r15
  _QWORD *v6; // rax
  __int64 v7; // rdx
  _BYTE *v8; // rbx
  __int64 result; // rax
  __int64 v10; // rsi
  __m128i *v11; // r15
  _QWORD *v12; // rax
  __int64 v13; // rdx

  v3 = *(__int64 **)(a1 + 72);
  v4 = v3[2];
  if ( *(_BYTE *)(a1 + 56) == 94 )
  {
    v5 = sub_73CA70(*(const __m128i **)(a2 + 120), *v3);
    v6 = sub_726700(4);
    v7 = *(_QWORD *)(a2 + 120);
    v6[7] = a2;
    *v6 = v7;
    v3[2] = (__int64)v6;
    v8 = sub_73DBF0(*(_BYTE *)(a1 + 56), (__int64)v5, (__int64)v3);
    result = sub_730580((__int64)v3, (__int64)v8);
  }
  else
  {
    if ( (unsigned int)sub_8D2E30(*v3) )
      v10 = sub_8D46C0(*v3);
    else
      v10 = sub_72C930();
    v11 = sub_73CA70(*(const __m128i **)(a2 + 120), v10);
    v12 = sub_726700(4);
    v13 = *(_QWORD *)(a2 + 120);
    v12[7] = a2;
    *v12 = v13;
    v3[2] = (__int64)v12;
    result = (__int64)sub_73DBF0(*(_BYTE *)(a1 + 56), (__int64)v11, (__int64)v3);
    *(_BYTE *)(result + 25) |= 1u;
    v8 = (_BYTE *)result;
  }
  v8[27] |= 2u;
  *((_QWORD *)v8 + 2) = v4;
  *(_QWORD *)(a1 + 72) = v8;
  *(_BYTE *)(a1 + 56) = 94;
  return result;
}
