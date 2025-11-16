// Function: sub_1E426C0
// Address: 0x1e426c0
//
_DWORD *__fastcall sub_1E426C0(_DWORD *a1, __int64 a2, _DWORD *a3)
{
  __int64 v5; // rsi
  __int64 v6; // rdx
  unsigned int v7; // esi
  _DWORD *v8; // rax
  unsigned int v10; // r9d
  unsigned int v11; // r10d
  int v12; // r11d

  v5 = a2 - (_QWORD)a1;
  v6 = 0xAAAAAAAAAAAAAAABLL * (v5 >> 5);
  if ( v5 > 0 )
  {
    v7 = a3[15];
    do
    {
      v8 = &a1[8 * (v6 >> 1) + 8 * (v6 & 0xFFFFFFFFFFFFFFFELL)];
      if ( v8[15] == v7 )
      {
        v10 = v8[18];
        if ( v10 && (v11 = a3[18], v10 != v11) && v11 )
        {
          if ( v10 >= v11 )
          {
            v6 >>= 1;
            continue;
          }
        }
        else
        {
          v12 = a3[16];
          if ( v8[16] == v12 )
          {
            if ( v8[17] <= a3[17] )
            {
              v6 >>= 1;
              continue;
            }
          }
          else if ( v8[16] >= v12 )
          {
LABEL_13:
            v6 >>= 1;
            continue;
          }
        }
      }
      else if ( v8[15] <= v7 )
      {
        goto LABEL_13;
      }
      a1 = v8 + 24;
      v6 = v6 - (v6 >> 1) - 1;
    }
    while ( v6 > 0 );
  }
  return a1;
}
