// Function: sub_1EE54A0
// Address: 0x1ee54a0
//
void __fastcall sub_1EE54A0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, int a5, __int64 a6, __int64 a7)
{
  __int64 v10; // r14
  __int16 v11; // r10
  __int64 v12; // rcx
  unsigned int v13; // eax
  int v14; // ebx
  unsigned int v15; // edi
  unsigned __int16 *v16; // rsi
  unsigned int v17; // edx
  int v18; // esi

  *(_QWORD *)(a7 + 4) = 0;
  if ( a2 )
  {
    v10 = a2;
    v11 = 1;
    v12 = 0;
    v13 = 0;
    do
    {
      v14 = *(_DWORD *)(a1 + 4 * v12);
      v15 = *(_DWORD *)(a3 + 4 * v12);
      if ( v14 != v15 )
      {
        if ( !*(_WORD *)(a7 + 4) && a5 != v13 )
        {
          while ( 1 )
          {
            v16 = (unsigned __int16 *)(a4 + 4LL * v13);
            v17 = *v16 - 1;
            if ( v17 >= (unsigned int)v12 )
              break;
            if ( a5 == ++v13 )
              goto LABEL_3;
          }
          if ( v17 == (_DWORD)v12 )
          {
            v18 = (__int16)v16[1];
            if ( (int)(v15 - v18) > 0 )
            {
              *(_WORD *)(a7 + 4) = v11;
              *(_WORD *)(a7 + 6) = v15 - v18;
            }
          }
        }
LABEL_3:
        if ( !*(_WORD *)(a7 + 8) && v15 > *(_DWORD *)(a6 + 4 * v12) )
        {
          *(_WORD *)(a7 + 8) = v11;
          *(_WORD *)(a7 + 10) = v15 - v14;
          if ( a5 == v13 || *(_WORD *)(a7 + 4) )
            break;
        }
      }
      ++v12;
      ++v11;
    }
    while ( v10 != v12 );
  }
}
