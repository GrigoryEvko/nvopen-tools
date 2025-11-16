// Function: sub_1993D70
// Address: 0x1993d70
//
float __fastcall sub_1993D70(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r8
  __int64 v4; // r10
  __int64 v5; // r9
  __int64 v6; // r11
  _QWORD *v7; // rdi
  _QWORD *v8; // rsi
  __int64 v9; // rax
  float v10; // xmm0_4
  __int64 v12[3]; // [rsp+8h] [rbp-18h] BYREF

  v2 = *(unsigned int *)(a1 + 752);
  v3 = *(_QWORD *)(a1 + 744);
  v4 = v3 + 96 * v2;
  if ( v4 == v3 )
  {
    v9 = *(unsigned int *)(a1 + 752);
  }
  else
  {
    v5 = a2;
    LODWORD(v6) = 0;
    do
    {
      v12[0] = v5;
      if ( v5 == *(_QWORD *)(v3 + 80)
        || (v7 = *(_QWORD **)(v3 + 32), v8 = &v7[*(unsigned int *)(v3 + 40)], v8 != sub_1993010(v7, (__int64)v8, v12)) )
      {
        v6 = (unsigned int)(v6 + 1);
      }
      v3 += 96;
    }
    while ( v3 != v4 );
    v9 = v2 - v6;
  }
  if ( v9 < 0 )
    v10 = (float)(v9 & 1 | (unsigned int)((unsigned __int64)v9 >> 1))
        + (float)(v9 & 1 | (unsigned int)((unsigned __int64)v9 >> 1));
  else
    v10 = (float)(int)v9;
  return v10 / (float)(int)v2;
}
