// Function: sub_131D720
// Address: 0x131d720
//
__int64 __fastcall sub_131D720(__int64 a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5, __int64 *a6, __int64 a7)
{
  unsigned int v7; // r10d
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rdx
  __int64 v10; // rbx
  __int64 *v11; // r14
  __int64 *v12; // r12
  __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // r8
  __int64 v16; // rcx

  if ( !a4 || !a5 || !a6 || !a7 )
    return 22;
  v7 = 22;
  v8 = a7 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (a7 & 7) == 0 )
  {
    v9 = v8 >> 3;
    if ( *a5 == 8 * ((v8 >> 3) + (v8 >> 2)) )
    {
      if ( v9 )
      {
        v10 = a4 + 8;
        v11 = &a6[v9];
        v12 = a6;
        do
        {
          v13 = *v12;
          v14 = v10 - 8;
          v15 = v10 + 8;
          v16 = v10;
          ++v12;
          v10 += 24;
          sub_134A520(a1, v13, v14, v16, v15);
        }
        while ( v12 != v11 );
      }
      return 0;
    }
  }
  return v7;
}
