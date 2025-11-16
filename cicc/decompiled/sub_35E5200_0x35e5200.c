// Function: sub_35E5200
// Address: 0x35e5200
//
void __fastcall sub_35E5200(__int64 a1, _DWORD *a2)
{
  _DWORD *v2; // rcx
  int v5; // edi
  _DWORD *v6; // rdx
  int v7; // r9d
  int v8; // r10d
  __int64 v9; // r11
  _DWORD *v10; // rax
  __int64 v11; // rsi
  unsigned __int64 v12; // rdx
  __int64 v13; // rsi
  _DWORD *i; // rax

  if ( (_DWORD *)a1 != a2 )
  {
    v2 = (_DWORD *)(a1 + 24);
    while ( a2 != v2 )
    {
      v5 = *v2;
      v6 = v2;
      v2 += 6;
      v7 = *(v2 - 5);
      v8 = *(v2 - 4);
      v9 = *((_QWORD *)v2 - 1);
      if ( v5 >= *(_DWORD *)a1 )
      {
        for ( i = v2 - 12; v5 < *i; i -= 6 )
        {
          *((_QWORD *)i + 5) = *((_QWORD *)i + 2);
          i[8] = i[2];
          i[7] = i[1];
          i[6] = *i;
          v6 = i;
        }
        *((_QWORD *)v6 + 2) = v9;
        v6[2] = v8;
        v6[1] = v7;
        *v6 = v5;
      }
      else
      {
        v10 = v2;
        v11 = (__int64)v6 - a1;
        v12 = 0xAAAAAAAAAAAAAAABLL * (((__int64)v6 - a1) >> 3);
        if ( v11 > 0 )
        {
          do
          {
            v13 = *((_QWORD *)v10 - 4);
            v10 -= 6;
            *((_QWORD *)v10 + 2) = v13;
            v10[2] = *(v10 - 4);
            v10[1] = *(v10 - 5);
            *v10 = *(v10 - 6);
            --v12;
          }
          while ( v12 );
        }
        *(_QWORD *)(a1 + 16) = v9;
        *(_DWORD *)(a1 + 8) = v8;
        *(_DWORD *)(a1 + 4) = v7;
        *(_DWORD *)a1 = v5;
      }
    }
  }
}
