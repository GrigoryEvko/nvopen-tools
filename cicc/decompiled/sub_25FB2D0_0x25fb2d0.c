// Function: sub_25FB2D0
// Address: 0x25fb2d0
//
void __fastcall sub_25FB2D0(unsigned __int64 *a1, unsigned __int64 *a2, unsigned __int64 *a3)
{
  __int64 v3; // rbx
  unsigned __int64 *v4; // r12
  unsigned __int64 *v5; // rdi
  __int64 v6; // r12
  __int64 v7; // rbx
  unsigned __int64 *v8; // r8
  unsigned __int64 *v9; // r15
  __int64 v10; // r14
  __int64 v11; // r12
  unsigned __int64 *v12; // rdi
  signed __int64 v13; // rax
  unsigned __int64 *v14; // r8
  char *v15; // r14
  __int64 v16; // r15
  __int64 v17; // rbx
  unsigned __int64 *v18; // rdi
  signed __int64 v19; // rax
  __int64 v20; // [rsp+0h] [rbp-60h]
  signed __int64 v21; // [rsp+0h] [rbp-60h]
  __int64 v24; // [rsp+18h] [rbp-48h]
  char *v25; // [rsp+20h] [rbp-40h]

  v3 = (char *)a2 - (char *)a1;
  v25 = (char *)a3 + (char *)a2 - (char *)a1;
  v24 = 0xAAAAAAAAAAAAAAABLL * (a2 - a1);
  if ( (char *)a2 - (char *)a1 <= 144 )
  {
    sub_25FA3C0(a1, a2);
  }
  else
  {
    v4 = a1;
    do
    {
      v5 = v4;
      v4 += 21;
      sub_25FA3C0(v5, v4);
    }
    while ( (char *)a2 - (char *)v4 > 144 );
    sub_25FA3C0(v4, a2);
    if ( v3 > 168 )
    {
      v6 = 7;
      while ( 1 )
      {
        v7 = 2 * v6;
        if ( v24 < 2 * v6 )
        {
          v8 = a3;
          v13 = v24;
          v9 = a1;
        }
        else
        {
          v8 = a3;
          v9 = a1;
          v20 = v6;
          v10 = 48 * v6;
          v11 = 24 * v6;
          do
          {
            v12 = v9;
            v9 = (unsigned __int64 *)((char *)v9 + v10);
            v8 = sub_25FAE10(
                   v12,
                   (unsigned __int64 *)((char *)v9 + v11 - v10),
                   (unsigned __int64 *)((char *)v9 + v11 - v10),
                   v9,
                   v8);
            v13 = 0xAAAAAAAAAAAAAAABLL * (a2 - v9);
          }
          while ( v7 <= v13 );
          v6 = v20;
        }
        if ( v6 <= v13 )
          v13 = v6;
        v6 *= 4;
        sub_25FAE10(v9, &v9[3 * v13], &v9[3 * v13], a2, v8);
        v14 = a1;
        if ( v24 < v6 )
          break;
        v21 = v7;
        v15 = (char *)a3;
        v16 = 24 * v6;
        v17 = 24 * v7;
        do
        {
          v18 = (unsigned __int64 *)v15;
          v15 += v16;
          v14 = sub_25FA960(v18, &v15[v17 - v16], (unsigned __int64 *)&v15[v17 - v16], v15, v14);
          v19 = 0xAAAAAAAAAAAAAAABLL * ((v25 - v15) >> 3);
        }
        while ( v6 <= v19 );
        if ( v19 > v21 )
          v19 = v21;
        sub_25FA960((unsigned __int64 *)v15, &v15[24 * v19], (unsigned __int64 *)&v15[24 * v19], v25, v14);
        if ( v24 <= v6 )
          return;
      }
      if ( v24 <= v7 )
        v7 = v24;
      sub_25FA960(a3, (char *)&a3[3 * v7], &a3[3 * v7], v25, a1);
    }
  }
}
