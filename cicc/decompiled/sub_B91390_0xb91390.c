// Function: sub_B91390
// Address: 0xb91390
//
__int64 __fastcall sub_B91390(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  int v5; // eax
  int v6; // ecx
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // rdi
  int v11; // eax
  int v12; // r8d

  v3 = *(_QWORD *)sub_BD5C60(a1, a2);
  v4 = *(_QWORD *)(v3 + 576);
  v5 = *(_DWORD *)(v3 + 592);
  if ( v5 )
  {
    v6 = v5 - 1;
    v7 = (v5 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v8 = (__int64 *)(v4 + 16LL * v7);
    v9 = *v8;
    if ( a1 == *v8 )
      return v8[1];
    v11 = 1;
    while ( v9 != -4096 )
    {
      v12 = v11 + 1;
      v7 = v6 & (v11 + v7);
      v8 = (__int64 *)(v4 + 16LL * v7);
      v9 = *v8;
      if ( a1 == *v8 )
        return v8[1];
      v11 = v12;
    }
  }
  return 0;
}
