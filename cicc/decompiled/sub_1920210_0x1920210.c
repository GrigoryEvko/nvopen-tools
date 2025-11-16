// Function: sub_1920210
// Address: 0x1920210
//
void __fastcall sub_1920210(__int64 a1, __int64 a2)
{
  unsigned int *v3; // rsi
  __int64 v4; // r9
  __int64 v5; // r11
  __int64 v6; // rcx
  unsigned int *i; // rax
  unsigned int v8; // edx
  __int64 v9; // r11
  unsigned int v10; // r8d
  unsigned int v11; // r10d
  unsigned int *v12; // rax
  __int64 v13; // r12
  __int64 v14; // r13
  __int64 v15; // r11
  unsigned __int64 v16; // rdx
  unsigned int v17; // ecx
  unsigned int v18; // edx

  if ( a1 != a2 )
  {
    v3 = (unsigned int *)(a1 + 24);
    if ( a2 != a1 + 24 )
    {
      v4 = a1 + 48;
      do
      {
        while ( 1 )
        {
          v10 = *v3;
          v11 = v3[1];
          v12 = v3;
          if ( *v3 < *(_DWORD *)a1 || v10 == *(_DWORD *)a1 && *(_DWORD *)(a1 + 4) > v11 )
            break;
          v5 = *((_QWORD *)v3 + 1);
          v6 = *((_QWORD *)v3 + 2);
          for ( i = v3; ; *((_QWORD *)i + 5) = *((_QWORD *)i + 2) )
          {
            v8 = *(i - 6);
            if ( v10 >= v8 && (v10 != v8 || *(i - 5) <= v11) )
              break;
            *i = v8;
            v18 = *(i - 5);
            i -= 6;
            i[7] = v18;
            *((_QWORD *)i + 4) = *((_QWORD *)i + 1);
          }
          *((_QWORD *)i + 1) = v5;
          v9 = v4;
          v3 += 6;
          v4 += 24;
          *i = v10;
          i[1] = v11;
          *((_QWORD *)i + 2) = v6;
          if ( a2 == v9 )
            return;
        }
        v13 = *((_QWORD *)v3 + 1);
        v14 = *((_QWORD *)v3 + 2);
        v15 = v4;
        v16 = 0xAAAAAAAAAAAAAAABLL * (((__int64)v3 - a1) >> 3);
        if ( (__int64)v3 - a1 > 0 )
        {
          do
          {
            v17 = *(v12 - 6);
            v12 -= 6;
            v12[6] = v17;
            v12[7] = v12[1];
            *((_QWORD *)v12 + 4) = *((_QWORD *)v12 + 1);
            *((_QWORD *)v12 + 5) = *((_QWORD *)v12 + 2);
            --v16;
          }
          while ( v16 );
        }
        *(_DWORD *)a1 = v10;
        v3 += 6;
        v4 += 24;
        *(_DWORD *)(a1 + 4) = v11;
        *(_QWORD *)(a1 + 8) = v13;
        *(_QWORD *)(a1 + 16) = v14;
      }
      while ( a2 != v15 );
    }
  }
}
