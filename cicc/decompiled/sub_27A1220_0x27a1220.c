// Function: sub_27A1220
// Address: 0x27a1220
//
void __fastcall sub_27A1220(__int64 a1, __int64 a2)
{
  int *v3; // rsi
  __int64 v4; // r9
  __int64 v5; // r11
  __int64 v6; // rcx
  unsigned int *i; // rax
  unsigned int v8; // edx
  __int64 v9; // r11
  unsigned int v10; // r8d
  unsigned __int64 v11; // r10
  int *v12; // rax
  __int64 v13; // r12
  __int64 v14; // r13
  __int64 v15; // r11
  __int64 v16; // rdx
  int v17; // ecx
  __int64 v18; // rdx

  if ( a1 != a2 )
  {
    v3 = (int *)(a1 + 32);
    if ( a2 != a1 + 32 )
    {
      v4 = a1 + 64;
      do
      {
        while ( 1 )
        {
          v10 = *v3;
          v11 = *((_QWORD *)v3 + 1);
          v12 = v3;
          if ( (unsigned int)*v3 < *(_DWORD *)a1 || v10 == *(_DWORD *)a1 && *(_QWORD *)(a1 + 8) > v11 )
            break;
          v5 = *((_QWORD *)v3 + 2);
          v6 = *((_QWORD *)v3 + 3);
          for ( i = (unsigned int *)v3; ; *((_QWORD *)i + 7) = *((_QWORD *)i + 3) )
          {
            v8 = *(i - 8);
            if ( v10 >= v8 && (v10 != v8 || *((_QWORD *)i - 3) <= v11) )
              break;
            *i = v8;
            v18 = *((_QWORD *)i - 3);
            i -= 8;
            *((_QWORD *)i + 5) = v18;
            *((_QWORD *)i + 6) = *((_QWORD *)i + 2);
          }
          *((_QWORD *)i + 2) = v5;
          v9 = v4;
          v3 += 8;
          v4 += 32;
          *i = v10;
          *((_QWORD *)i + 1) = v11;
          *((_QWORD *)i + 3) = v6;
          if ( a2 == v9 )
            return;
        }
        v13 = *((_QWORD *)v3 + 2);
        v14 = *((_QWORD *)v3 + 3);
        v15 = v4;
        v16 = ((__int64)v3 - a1) >> 5;
        if ( (__int64)v3 - a1 > 0 )
        {
          do
          {
            v17 = *(v12 - 8);
            v12 -= 8;
            v12[8] = v17;
            *((_QWORD *)v12 + 5) = *((_QWORD *)v12 + 1);
            *((_QWORD *)v12 + 6) = *((_QWORD *)v12 + 2);
            *((_QWORD *)v12 + 7) = *((_QWORD *)v12 + 3);
            --v16;
          }
          while ( v16 );
        }
        *(_DWORD *)a1 = v10;
        v3 += 8;
        v4 += 32;
        *(_QWORD *)(a1 + 8) = v11;
        *(_QWORD *)(a1 + 16) = v13;
        *(_QWORD *)(a1 + 24) = v14;
      }
      while ( a2 != v15 );
    }
  }
}
