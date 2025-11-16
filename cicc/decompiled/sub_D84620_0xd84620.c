// Function: sub_D84620
// Address: 0xd84620
//
unsigned __int64 __fastcall sub_D84620(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // r13
  __int64 v3; // rax
  __int64 v4; // rax
  bool v5; // cf
  bool v6; // zf
  unsigned __int64 result; // rax
  __int64 v8; // rax
  double v9; // xmm0_8
  double v10; // xmm0_8
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rdx
  __int64 v13; // rdx

  v1 = *(_QWORD *)(a1 + 8) + 8LL;
  v2 = sub_EF9A70(v1, SLODWORD(qword_4F8AD68[8]));
  v3 = sub_EF9B00(v1);
  *(_BYTE *)(a1 + 24) = 1;
  *(_QWORD *)(a1 + 16) = v3;
  v4 = sub_EF9C00(v1);
  *(_BYTE *)(a1 + 40) = 1;
  *(_QWORD *)(a1 + 32) = v4;
  if ( (unsigned __int8)sub_D845F0(a1) && LOBYTE(qword_4F87DA8[8]) )
  {
    v8 = *(_QWORD *)(v2 + 16);
    if ( v8 < 0 )
    {
      v13 = *(_QWORD *)(v2 + 16) & 1LL | (*(_QWORD *)(v2 + 16) >> 1);
      v9 = (double)(int)v13 + (double)(int)v13;
    }
    else
    {
      v9 = (double)(int)v8;
    }
    v10 = v9 * *(double *)(*(_QWORD *)(a1 + 8) + 80LL) * *(double *)&qword_4F87D08;
    if ( v10 >= 9.223372036854776e18 )
      result = (unsigned int)(int)(v10 - 9.223372036854776e18) ^ 0x8000000000000000LL;
    else
      result = (unsigned int)(int)v10;
    v11 = LODWORD(qword_4F8ABA8[8]);
    *(_BYTE *)(a1 + 49) = 1;
    *(_BYTE *)(a1 + 48) = v11 < result;
    v12 = LODWORD(qword_4F8AAC8[8]);
    *(_BYTE *)(a1 + 51) = 1;
    *(_BYTE *)(a1 + 50) = v12 < result;
  }
  else
  {
    v5 = *(_QWORD *)(v2 + 16) < (unsigned __int64)LODWORD(qword_4F8ABA8[8]);
    v6 = *(_QWORD *)(v2 + 16) == LODWORD(qword_4F8ABA8[8]);
    *(_BYTE *)(a1 + 49) = 1;
    *(_BYTE *)(a1 + 48) = !v5 && !v6;
    result = LODWORD(qword_4F8AAC8[8]);
    v5 = *(_QWORD *)(v2 + 16) < (unsigned __int64)LODWORD(qword_4F8AAC8[8]);
    v6 = *(_QWORD *)(v2 + 16) == LODWORD(qword_4F8AAC8[8]);
    *(_BYTE *)(a1 + 51) = 1;
    *(_BYTE *)(a1 + 50) = !v5 && !v6;
  }
  return result;
}
