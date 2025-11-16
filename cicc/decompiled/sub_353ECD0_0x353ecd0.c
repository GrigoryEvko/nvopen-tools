// Function: sub_353ECD0
// Address: 0x353ecd0
//
_DWORD *__fastcall sub_353ECD0(_DWORD *a1, __int64 a2, _DWORD *a3)
{
  __int64 v5; // rsi
  __int64 v6; // rdx
  unsigned int v7; // esi
  _DWORD *v8; // rcx
  unsigned int v10; // r9d
  unsigned int v11; // r10d
  int v12; // r11d

  v5 = a2 - (_QWORD)a1;
  v6 = 0x2E8BA2E8BA2E8BA3LL * (v5 >> 3);
  if ( v5 > 0 )
  {
    v7 = a3[13];
    do
    {
      v8 = &a1[22 * (v6 >> 1)];
      if ( v8[13] == v7 )
      {
        v10 = v8[16];
        if ( v10 && (v11 = a3[16]) != 0 && v10 != v11 )
        {
          if ( v10 >= v11 )
          {
            v6 >>= 1;
            continue;
          }
        }
        else
        {
          v12 = a3[14];
          if ( v8[14] == v12 )
          {
            if ( v8[15] <= a3[15] )
            {
              v6 >>= 1;
              continue;
            }
          }
          else if ( v8[14] >= v12 )
          {
LABEL_13:
            v6 >>= 1;
            continue;
          }
        }
      }
      else if ( v8[13] <= v7 )
      {
        goto LABEL_13;
      }
      a1 = v8 + 22;
      v6 = v6 - (v6 >> 1) - 1;
    }
    while ( v6 > 0 );
  }
  return a1;
}
