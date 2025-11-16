// Function: sub_1DE3990
// Address: 0x1de3990
//
void __fastcall sub_1DE3990(__int64 *src, __int64 *a2, __int64 *a3)
{
  __int64 v3; // rax
  __int64 *v4; // rbx
  __int64 v5; // rdi
  __int64 v6; // r8
  __int64 *v7; // rdx
  __int64 i; // rcx
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 *v12; // rcx
  __int64 *v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // r9
  __int64 v16; // r10
  __int64 *v17; // rax
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // r12
  __int64 v21; // r13
  __int64 *v22; // rax
  __int64 v23; // r12

  if ( src != a2 && a2 != a3 )
  {
    v3 = a3 - src;
    v4 = src;
    v5 = a2 - src;
    if ( v5 == v3 - v5 )
    {
      v17 = a2;
      do
      {
        v18 = *v17;
        v19 = *v4++;
        ++v17;
        *(v4 - 1) = v18;
        *(v17 - 1) = v19;
      }
      while ( a2 != v4 );
    }
    else
    {
      while ( 1 )
      {
        v6 = v3 - v5;
        if ( v5 < v3 - v5 )
          break;
LABEL_12:
        v11 = v3;
        if ( v6 == 1 )
        {
          v22 = &v4[v11 - 1];
          v23 = *v22;
          if ( v4 != v22 )
            memmove(v4 + 1, v4, v11 * 8 - 8);
          *v4 = v23;
          return;
        }
        v12 = &v4[v11];
        v4 = &v12[-v6];
        if ( v5 > 0 )
        {
          v13 = &v12[-v6];
          v14 = 0;
          do
          {
            v15 = *(v13 - 1);
            v16 = *(v12 - 1);
            ++v14;
            --v13;
            --v12;
            *v13 = v16;
            *v12 = v15;
          }
          while ( v5 != v14 );
          v4 -= v5;
        }
        v5 = v3 % v6;
        if ( !(v3 % v6) )
          return;
        v3 = v6;
      }
      while ( v5 != 1 )
      {
        v7 = &v4[v5];
        if ( v6 > 0 )
        {
          for ( i = 0; i != v6; ++i )
          {
            v9 = v4[i];
            v4[i] = v7[i];
            v7[i] = v9;
          }
          v4 += v6;
        }
        v10 = v3 % v5;
        if ( !(v3 % v5) )
          return;
        v3 = v5;
        v5 -= v10;
        v6 = v3 - v5;
        if ( v5 >= v3 - v5 )
          goto LABEL_12;
      }
      v20 = v3;
      v21 = *v4;
      if ( v4 + 1 != &v4[v3] )
        memmove(v4, v4 + 1, v20 * 8 - 8);
      v4[v20 - 1] = v21;
    }
  }
}
