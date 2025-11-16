// Function: sub_1E42610
// Address: 0x1e42610
//
_DWORD *__fastcall sub_1E42610(_DWORD *a1, __int64 a2, _DWORD *a3)
{
  _DWORD *v5; // r8
  __int64 v6; // rax
  __int64 v7; // rdx
  unsigned int v8; // edi
  _DWORD *v9; // rax
  unsigned int v11; // r9d
  unsigned int v12; // r10d
  int v13; // r11d

  v5 = a1;
  v6 = a2 - (_QWORD)a1;
  v7 = 0xAAAAAAAAAAAAAAABLL * (v6 >> 5);
  if ( v6 > 0 )
  {
    v8 = a3[15];
    do
    {
      v9 = &v5[8 * (v7 >> 1) + 8 * (v7 & 0xFFFFFFFFFFFFFFFELL)];
      if ( v8 == v9[15] )
      {
        v11 = a3[18];
        if ( v11 && (v12 = v9[18], v11 != v12) && v12 )
        {
          if ( v11 < v12 )
          {
            v7 >>= 1;
            continue;
          }
        }
        else
        {
          v13 = v9[16];
          if ( a3[16] == v13 )
          {
            if ( a3[17] > v9[17] )
            {
              v7 >>= 1;
              continue;
            }
          }
          else if ( a3[16] < v13 )
          {
LABEL_13:
            v7 >>= 1;
            continue;
          }
        }
      }
      else if ( v8 > v9[15] )
      {
        goto LABEL_13;
      }
      v5 = v9 + 24;
      v7 = v7 - (v7 >> 1) - 1;
    }
    while ( v7 > 0 );
  }
  return v5;
}
