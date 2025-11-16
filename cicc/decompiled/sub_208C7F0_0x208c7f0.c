// Function: sub_208C7F0
// Address: 0x208c7f0
//
void __fastcall sub_208C7F0(__int64 a1, __int64 *a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rsi
  unsigned int v9; // ecx
  __int64 v10; // rdx
  __int64 *v11; // r8
  int v12; // edx
  int v13; // r9d

  if ( !(unsigned __int8)sub_1642FB0(*a2) )
  {
    v6 = *(_QWORD *)(a1 + 712);
    v7 = *(unsigned int *)(v6 + 232);
    if ( (_DWORD)v7 )
    {
      v8 = *(_QWORD *)(v6 + 216);
      v9 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v10 = v8 + 16LL * v9;
      v11 = *(__int64 **)v10;
      if ( a2 == *(__int64 **)v10 )
      {
LABEL_5:
        if ( v10 != v8 + 16 * v7 )
          sub_208C270(a1, a2, *(_DWORD *)(v10 + 8), a3, a4, a5);
      }
      else
      {
        v12 = 1;
        while ( v11 != (__int64 *)-8LL )
        {
          v13 = v12 + 1;
          v9 = (v7 - 1) & (v12 + v9);
          v10 = v8 + 16LL * v9;
          v11 = *(__int64 **)v10;
          if ( a2 == *(__int64 **)v10 )
            goto LABEL_5;
          v12 = v13;
        }
      }
    }
  }
}
