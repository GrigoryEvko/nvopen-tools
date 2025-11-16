// Function: sub_276E740
// Address: 0x276e740
//
__int64 *__fastcall sub_276E740(__int64 *a1, __int64 a2, char *a3, __int64 *a4)
{
  char *v5; // rsi
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rbx
  __int64 v13; // rdi
  char *v14; // r15
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 *v18; // rcx
  __int64 v19; // rax

  v5 = a3;
  v6 = (__int64)&a3[-a2];
  v7 = (__int64)&a3[-a2] >> 3;
  v10 = *a4;
  v11 = a4[1];
  if ( v6 > 0 )
  {
    while ( 1 )
    {
      v15 = (v10 - v11) >> 3;
      if ( v10 == v11 )
      {
        v12 = 64;
        v13 = *(_QWORD *)(a4[3] - 8) + 512LL;
      }
      else
      {
        v12 = (v10 - v11) >> 3;
        v13 = v10;
      }
      if ( v12 > v7 )
        v12 = v7;
      v14 = &v5[-8 * v12];
      if ( v14 != v5 )
      {
        memmove((void *)(-8 * v12 + v13), v14, 8 * v12);
        v10 = *a4;
        v11 = a4[1];
        v15 = (*a4 - v11) >> 3;
      }
      v16 = v15 - v12;
      if ( v16 < 0 )
      {
        v17 = ~((unsigned __int64)~v16 >> 6);
      }
      else
      {
        if ( v16 <= 63 )
        {
          v7 -= v12;
          v10 -= 8 * v12;
          *a4 = v10;
          if ( v7 <= 0 )
            break;
          goto LABEL_11;
        }
        v17 = v16 >> 6;
      }
      v7 -= v12;
      v18 = (__int64 *)(a4[3] + 8 * v17);
      a4[3] = (__int64)v18;
      v11 = *v18;
      v10 = v11 + 8 * (v16 - (v17 << 6));
      a4[1] = v11;
      a4[2] = v11 + 512;
      *a4 = v10;
      if ( v7 <= 0 )
        break;
LABEL_11:
      v5 -= 8 * v12;
    }
  }
  v19 = a4[2];
  *a1 = v10;
  a1[1] = v11;
  a1[2] = v19;
  a1[3] = a4[3];
  return a1;
}
