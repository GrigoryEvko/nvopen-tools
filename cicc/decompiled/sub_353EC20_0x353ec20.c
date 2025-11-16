// Function: sub_353EC20
// Address: 0x353ec20
//
_DWORD *__fastcall sub_353EC20(_DWORD *a1, __int64 a2, _DWORD *a3)
{
  _DWORD *v5; // r8
  __int64 v6; // rax
  __int64 v7; // rdx
  unsigned int v8; // edi
  _DWORD *v9; // rcx
  unsigned int v11; // r9d
  unsigned int v12; // r10d
  int v13; // r11d

  v5 = a1;
  v6 = a2 - (_QWORD)a1;
  v7 = 0x2E8BA2E8BA2E8BA3LL * (v6 >> 3);
  if ( v6 > 0 )
  {
    v8 = a3[13];
    do
    {
      v9 = &v5[22 * (v7 >> 1)];
      if ( v8 == v9[13] )
      {
        v11 = a3[16];
        if ( v11 && (v12 = v9[16]) != 0 && v11 != v12 )
        {
          if ( v11 < v12 )
          {
            v7 >>= 1;
            continue;
          }
        }
        else
        {
          v13 = v9[14];
          if ( a3[14] == v13 )
          {
            if ( a3[15] > v9[15] )
            {
              v7 >>= 1;
              continue;
            }
          }
          else if ( a3[14] < v13 )
          {
LABEL_13:
            v7 >>= 1;
            continue;
          }
        }
      }
      else if ( v8 > v9[13] )
      {
        goto LABEL_13;
      }
      v5 = v9 + 22;
      v7 = v7 - (v7 >> 1) - 1;
    }
    while ( v7 > 0 );
  }
  return v5;
}
