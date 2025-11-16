// Function: sub_B91FC0
// Address: 0xb91fc0
//
__int64 *__fastcall sub_B91FC0(__int64 *a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rsi
  unsigned int v7; // eax
  __int64 *v8; // r13
  __int64 v9; // rcx
  __int64 *v10; // r13
  int v11; // r8d

  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  a1[3] = 0;
  if ( (*(_BYTE *)(a2 + 7) & 0x20) == 0 )
    return a1;
  v4 = *(_QWORD *)sub_BD5C60(a2, a2);
  v5 = *(unsigned int *)(v4 + 3248);
  v6 = *(_QWORD *)(v4 + 3232);
  if ( !(_DWORD)v5 )
    goto LABEL_6;
  v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v6 + 40LL * v7);
  v9 = *v8;
  if ( a2 != *v8 )
  {
    v11 = 1;
    while ( v9 != -4096 )
    {
      v7 = (v5 - 1) & (v11 + v7);
      v8 = (__int64 *)(v6 + 40LL * v7);
      v9 = *v8;
      if ( a2 == *v8 )
        goto LABEL_5;
      ++v11;
    }
LABEL_6:
    v8 = (__int64 *)(v6 + 40 * v5);
  }
LABEL_5:
  v10 = v8 + 1;
  *a1 = sub_B91B30(v10, 1);
  a1[1] = sub_B91B30(v10, 5);
  a1[2] = sub_B91B30(v10, 7);
  a1[3] = sub_B91B30(v10, 8);
  return a1;
}
