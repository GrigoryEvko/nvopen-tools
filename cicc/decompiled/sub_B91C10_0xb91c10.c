// Function: sub_B91C10
// Address: 0xb91c10
//
__int64 __fastcall sub_B91C10(__int64 a1, __int64 a2)
{
  int v2; // r12d
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rsi
  unsigned int v7; // eax
  __int64 *v8; // rdi
  __int64 v9; // rcx
  int v11; // edi
  int v12; // r9d

  v2 = a2;
  v4 = *(_QWORD *)sub_BD5C60(a1, a2);
  v5 = *(unsigned int *)(v4 + 3248);
  v6 = *(_QWORD *)(v4 + 3232);
  if ( (_DWORD)v5 )
  {
    v7 = (v5 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v8 = (__int64 *)(v6 + 40LL * v7);
    v9 = *v8;
    if ( a1 == *v8 )
      return sub_B91B30(v8 + 1, v2);
    v11 = 1;
    while ( v9 != -4096 )
    {
      v12 = v11 + 1;
      v7 = (v5 - 1) & (v11 + v7);
      v8 = (__int64 *)(v6 + 40LL * v7);
      v9 = *v8;
      if ( a1 == *v8 )
        return sub_B91B30(v8 + 1, v2);
      v11 = v12;
    }
  }
  return sub_B91B30((__int64 *)(v6 + 40 * v5 + 8), v2);
}
