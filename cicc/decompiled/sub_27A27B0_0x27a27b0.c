// Function: sub_27A27B0
// Address: 0x27a27b0
//
__int64 __fastcall sub_27A27B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v6; // eax
  unsigned int v7; // r8d
  __int64 v8; // r13
  int v9; // r14d
  unsigned int v10; // ebx
  unsigned int v11; // r15d
  int v13; // eax
  int v15; // [rsp+14h] [rbp-3Ch]
  __int64 v16; // [rsp+18h] [rbp-38h]

  v16 = a2;
  v6 = sub_B46E30(a4);
  v7 = 0;
  if ( v6 <= (unsigned int)((a3 - a2) >> 5) )
  {
    if ( a2 != a3 )
    {
      do
      {
        v8 = *(_QWORD *)(v16 + 16);
        if ( !a4 )
          return 0;
        v15 = sub_B46E30(a4);
        v9 = v15 >> 2;
        if ( v15 >> 2 > 0 )
        {
          v10 = 0;
          while ( v8 != sub_B46EC0(a4, v10) )
          {
            v11 = v10 + 1;
            if ( v8 == sub_B46EC0(a4, v10 + 1)
              || (v11 = v10 + 2, v8 == sub_B46EC0(a4, v10 + 2))
              || (v11 = v10 + 3, v8 == sub_B46EC0(a4, v10 + 3)) )
            {
              v10 = v11;
              goto LABEL_11;
            }
            v10 += 4;
            if ( !--v9 )
            {
              v13 = v15 - v10;
              goto LABEL_17;
            }
          }
          goto LABEL_11;
        }
        v13 = v15;
        v10 = 0;
LABEL_17:
        switch ( v13 )
        {
          case 2:
            if ( v8 == sub_B46EC0(a4, v10) )
              goto LABEL_11;
            break;
          case 3:
            if ( v8 == sub_B46EC0(a4, v10) )
              goto LABEL_11;
            if ( v8 == sub_B46EC0(a4, ++v10) )
              goto LABEL_11;
            break;
          case 1:
            goto LABEL_20;
          default:
            return 0;
        }
        ++v10;
LABEL_20:
        if ( v8 != sub_B46EC0(a4, v10) )
          return 0;
LABEL_11:
        if ( v10 == v15 )
          return 0;
        v16 += 32;
      }
      while ( v16 != a3 );
    }
    return 1;
  }
  return v7;
}
