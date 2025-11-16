// Function: sub_1371BB0
// Address: 0x1371bb0
//
void __fastcall sub_1371BB0(__int64 a1, int a2)
{
  unsigned __int64 v3; // rdi
  int v4; // edx
  __int16 v5; // ax
  int v6; // r13d
  __int16 v7; // ax
  unsigned __int64 v8; // rax
  int v9; // ebx
  int v10; // edx
  unsigned __int64 v11; // rdx
  int v12; // edx
  int v13; // ebx

  if ( a2 )
  {
    v3 = *(_QWORD *)a1;
    if ( v3 )
    {
      v4 = *(__int16 *)(a1 + 8);
      v5 = *(_WORD *)(a1 + 8);
      if ( a2 < 0 )
      {
        v12 = v4 + 16382;
        if ( -a2 > v12 )
        {
          v13 = -a2 - v12;
          *(_WORD *)(a1 + 8) = v5 - v12;
          if ( v13 > 63 )
          {
            *(_QWORD *)a1 = 0;
            *(_WORD *)(a1 + 8) = 0;
          }
          else
          {
            *(_QWORD *)a1 = v3 >> v13;
          }
        }
        else
        {
          *(_WORD *)(a1 + 8) = v5 + a2;
        }
      }
      else
      {
        v6 = 0x3FFF - v4;
        if ( 0x3FFF - v4 < a2 )
        {
          v7 = v6 + v5;
          *(_WORD *)(a1 + 8) = v7;
          if ( (unsigned int)sub_1371720(v3, v7, 0xFFFFFFFFFFFFFFFFLL, 0x3FFF) )
          {
            v8 = *(_QWORD *)a1;
            v9 = a2 - v6;
            v10 = 64;
            if ( *(_QWORD *)a1 )
            {
              _BitScanReverse64(&v11, v8);
              v10 = v11 ^ 0x3F;
            }
            if ( v10 < v9 )
            {
              *(_QWORD *)a1 = -1;
              *(_WORD *)(a1 + 8) = 0x3FFF;
            }
            else
            {
              *(_QWORD *)a1 = v8 << v9;
            }
          }
        }
        else
        {
          *(_WORD *)(a1 + 8) = a2 + v5;
        }
      }
    }
  }
}
