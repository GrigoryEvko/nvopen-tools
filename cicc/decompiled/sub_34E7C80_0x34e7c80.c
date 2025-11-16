// Function: sub_34E7C80
// Address: 0x34e7c80
//
void __fastcall sub_34E7C80(unsigned __int64 *a1, unsigned __int64 *a2, unsigned __int64 *a3, __int64 a4)
{
  __int64 v5; // rbx
  unsigned __int64 *v6; // r12
  unsigned __int64 *v7; // rdi
  __int64 v8; // rbx
  __int64 v9; // r13
  __int64 v10; // r12
  unsigned __int64 *v11; // r8
  unsigned __int64 *v12; // r14
  __int64 v13; // rax
  __int64 v14; // r15
  __int64 v15; // rbx
  __int64 v16; // r13
  __int64 v17; // r15
  unsigned __int64 *v18; // rdi
  __int64 v19; // rax
  unsigned __int64 *v20; // r8
  unsigned __int64 *v21; // r14
  unsigned __int64 *v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // [rsp+0h] [rbp-60h]
  __int64 v28; // [rsp+18h] [rbp-48h]
  unsigned __int64 *v29; // [rsp+28h] [rbp-38h]

  v5 = (char *)a2 - (char *)a1;
  v28 = a2 - a1;
  v29 = (unsigned __int64 *)((char *)a3 + (char *)a2 - (char *)a1);
  if ( (char *)a2 - (char *)a1 <= 48 )
  {
    sub_34E7A40(a1, a2);
  }
  else
  {
    v6 = a1;
    do
    {
      v7 = v6;
      v6 += 7;
      sub_34E7A40(v7, v6);
    }
    while ( (char *)a2 - (char *)v6 > 48 );
    sub_34E7A40(v6, a2);
    if ( v5 > 56 )
    {
      v8 = 7;
      v9 = a4;
      while ( 1 )
      {
        v10 = 2 * v8;
        if ( v28 < 2 * v8 )
        {
          v11 = a3;
          v19 = v28;
          v12 = a1;
        }
        else
        {
          v11 = a3;
          v12 = a1;
          v25 = v8;
          v13 = 16 * v8;
          v14 = 8 * v8;
          v15 = v9;
          v16 = v14 - v13;
          v17 = v13;
          do
          {
            v18 = v12;
            v12 = (unsigned __int64 *)((char *)v12 + v17);
            v11 = sub_34E7670(
                    v18,
                    (unsigned __int64 *)((char *)v12 + v16),
                    (unsigned __int64 *)((char *)v12 + v16),
                    v12,
                    v11);
            v19 = a2 - v12;
          }
          while ( v10 <= v19 );
          v9 = v15;
          v8 = v25;
        }
        if ( v8 <= v19 )
          v19 = v8;
        v8 *= 4;
        sub_34E7670(v12, &v12[v19], &v12[v19], a2, v11);
        v20 = a1;
        if ( v28 < v8 )
          break;
        v21 = a3;
        do
        {
          v22 = v21;
          v21 += v8;
          v20 = sub_34E7860(
                  v22,
                  (unsigned __int64 *)((char *)v21 + 8 * v10 - 8 * v8),
                  (unsigned __int64 *)((char *)v21 + 8 * v10 - 8 * v8),
                  v21,
                  v20);
          v23 = v29 - v21;
        }
        while ( v8 <= v23 );
        if ( v23 > v10 )
          v23 = v10;
        sub_34E7860(v21, &v21[v23], &v21[v23], v29, v20);
        if ( v28 <= v8 )
          return;
      }
      v24 = v10;
      if ( v28 <= v10 )
        v24 = v28;
      sub_34E7860(a3, &a3[v24], &a3[v24], v29, a1);
    }
  }
}
