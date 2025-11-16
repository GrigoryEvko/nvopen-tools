// Function: sub_18FBDC0
// Address: 0x18fbdc0
//
void __fastcall sub_18FBDC0(char *src, char *a2, char *a3)
{
  __int64 v3; // r8
  __int64 v4; // rax
  char *v5; // rbx
  __int64 v6; // r9
  char *v7; // rdx
  char *v8; // r10
  char *v9; // rcx
  char v10; // si
  char v11; // di
  __int64 v12; // rdx
  char *v13; // rdx
  char *v14; // rcx
  char *v15; // r10
  char v16; // si
  char v17; // di
  char *v18; // rax
  char v19; // cl
  char v20; // dl
  size_t v21; // rax
  char v22; // r12
  size_t v23; // r13
  char v24; // r12
  size_t v25; // rax

  if ( src != a2 && a2 != a3 )
  {
    v3 = a2 - src;
    v4 = a3 - src;
    v5 = src;
    if ( a2 - src == a3 - a2 )
    {
      v18 = a2;
      do
      {
        v19 = *v18;
        v20 = *v5++;
        ++v18;
        *(v5 - 1) = v19;
        *(v18 - 1) = v20;
      }
      while ( a2 != v5 );
    }
    else
    {
      while ( 1 )
      {
        v6 = v4 - v3;
        if ( v3 < v4 - v3 )
          break;
LABEL_12:
        if ( v6 == 1 )
        {
          v24 = v5[v4 - 1];
          v25 = v4 - 1;
          if ( v25 )
            memmove(v5 + 1, v5, v25);
          *v5 = v24;
          return;
        }
        v13 = &v5[v4];
        v5 = &v5[v4 - v6];
        if ( v3 > 0 )
        {
          v14 = v5;
          v15 = &v13[-v3];
          do
          {
            v16 = *(v14 - 1);
            v17 = *--v13;
            *--v14 = v17;
            *v13 = v16;
          }
          while ( v15 != v13 );
          v5 -= v3;
        }
        v3 = v4 % v6;
        if ( !(v4 % v6) )
          return;
        v4 = v6;
      }
      while ( v3 != 1 )
      {
        v7 = &v5[v3];
        if ( v6 > 0 )
        {
          v8 = &v7[v6];
          v9 = v5;
          do
          {
            v10 = *v9;
            v11 = *v7++;
            *v9++ = v11;
            *(v7 - 1) = v10;
          }
          while ( v7 != v8 );
          v5 += v6;
        }
        v12 = v4 % v3;
        if ( !(v4 % v3) )
          return;
        v4 = v3;
        v3 -= v12;
        v6 = v4 - v3;
        if ( v3 >= v4 - v3 )
          goto LABEL_12;
      }
      v21 = v4 - 1;
      v22 = *v5;
      v23 = v21;
      if ( v21 )
        memmove(v5, v5 + 1, v21);
      v5[v23] = v22;
    }
  }
}
