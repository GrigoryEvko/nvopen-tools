// Function: sub_1E12BC0
// Address: 0x1e12bc0
//
__int64 __fastcall sub_1E12BC0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // r13
  __int64 result; // rax
  __int64 v5; // r14
  __int64 v6; // r15
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rsi
  _BYTE *v12; // rdx
  char v13; // cl

  if ( *(_QWORD *)(a1 + 248) && !(*(unsigned __int8 (__fastcall **)(__int64))(a1 + 256))(a1 + 232) )
    return 0;
  v2 = *(_QWORD *)(a2 + 328);
  v3 = a2 + 320;
  result = 0;
  if ( v2 == a2 + 320 )
    return 0;
  do
  {
    v5 = *(_QWORD *)(v2 + 32);
    v6 = v2 + 24;
    if ( v5 != v2 + 24 )
    {
      while ( 1 )
      {
        v7 = *(_QWORD *)(v5 + 8);
        if ( **(_WORD **)(v5 + 16) == 16 )
        {
          for ( ; v6 != v7; v7 = *(_QWORD *)(v7 + 8) )
          {
            if ( (*(_BYTE *)(v7 + 46) & 4) == 0 )
              break;
            sub_1E16420(v7);
            v8 = *(unsigned int *)(v7 + 40);
            if ( (_DWORD)v8 )
            {
              v9 = 5 * v8;
              v10 = 0;
              v11 = 8 * v9;
              do
              {
                v12 = (_BYTE *)(v10 + *(_QWORD *)(v7 + 32));
                if ( !*v12 )
                {
                  v13 = v12[4];
                  if ( (v13 & 2) != 0 )
                    v12[4] = v13 & 0xFD;
                }
                v10 += 40;
              }
              while ( v11 != v10 );
            }
          }
          sub_1E16240(v5);
          result = 1;
        }
        if ( v7 == v6 )
          break;
        v5 = v7;
      }
    }
    v2 = *(_QWORD *)(v2 + 8);
  }
  while ( v3 != v2 );
  return result;
}
