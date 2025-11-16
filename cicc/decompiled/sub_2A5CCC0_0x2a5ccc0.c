// Function: sub_2A5CCC0
// Address: 0x2a5ccc0
//
bool __fastcall sub_2A5CCC0(__int64 a1)
{
  _QWORD *v2; // rdx
  _QWORD *v3; // rax
  __int64 *v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r15
  __int64 v7; // rdx
  __int64 v8; // rbx
  __int64 v9; // rsi
  unsigned __int64 v10; // rbx
  __int64 *v11; // rax
  __int64 v12; // rdi
  _QWORD *v13; // rax
  _QWORD *v14; // r8
  _QWORD *v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 *v20; // rdi
  __int64 v21; // rdx
  __int64 v23; // [rsp+8h] [rbp-98h]
  __int64 v24; // [rsp+18h] [rbp-88h] BYREF
  __int64 v25[2]; // [rsp+20h] [rbp-80h] BYREF
  __int64 *v26; // [rsp+30h] [rbp-70h]
  __int64 *v27; // [rsp+38h] [rbp-68h]
  __int64 v28; // [rsp+40h] [rbp-60h]
  __int64 **v29; // [rsp+48h] [rbp-58h]
  __int64 *v30; // [rsp+50h] [rbp-50h]
  __int64 v31; // [rsp+58h] [rbp-48h]
  __int64 v32; // [rsp+60h] [rbp-40h]
  __int64 v33; // [rsp+68h] [rbp-38h]

  v2 = *(_QWORD **)(a1 + 8);
  v3 = *(_QWORD **)a1;
  if ( *(_QWORD **)a1 != v2 )
  {
    do
    {
      *v3 = 0x4000000000000LL;
      v3 += 9;
      *(v3 - 8) = -1;
      *(v3 - 7) = -1;
      *((_BYTE *)v3 - 48) = 0;
    }
    while ( v2 != v3 );
  }
  v25[0] = 0;
  v25[1] = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  sub_2A5CBD0(v25, 0);
  v4 = v30;
  if ( v30 == (__int64 *)(v32 - 8) )
  {
    sub_FE0450(v25, (_QWORD *)(a1 + 48));
    v5 = *(_QWORD *)(a1 + 48);
  }
  else
  {
    v5 = *(_QWORD *)(a1 + 48);
    if ( v30 )
    {
      *v30 = v5;
      v4 = v30;
    }
    v30 = v4 + 1;
  }
  *(_QWORD *)(*(_QWORD *)a1 + 72 * v5) = 0;
  *(_BYTE *)(*(_QWORD *)a1 + 72LL * *(_QWORD *)(a1 + 48) + 24) = 1;
  while ( v26 != v30 )
  {
    v6 = *v26;
    if ( v26 == (__int64 *)(v28 - 8) )
    {
      j_j___libc_free_0((unsigned __int64)v27);
      v21 = (__int64)(*++v29 + 64);
      v27 = *v29;
      v28 = v21;
      v26 = v27;
    }
    else
    {
      ++v26;
    }
    v7 = 72 * v6;
    *(_BYTE *)(*(_QWORD *)a1 + 72 * v6 + 24) = 0;
    v8 = *(_QWORD *)(*(_QWORD *)a1 + 72LL * *(_QWORD *)(a1 + 56));
    if ( !**(_BYTE **)(a1 + 88) && !v8 )
      goto LABEL_28;
    if ( *(_QWORD *)(*(_QWORD *)a1 + 72 * v6) <= v8 )
    {
      v9 = *(_QWORD *)(a1 + 24);
      v10 = 0;
      v11 = (__int64 *)(v9 + 24 * v6);
      v12 = *v11;
      if ( *v11 != v11[1] )
      {
        do
        {
          v13 = (_QWORD *)(v12 + 56 * v10);
          if ( v13[2] < v13[1] )
          {
            v14 = *(_QWORD **)a1;
            v24 = v13[3];
            v15 = &v14[9 * v24];
            v16 = *(_QWORD *)((char *)v14 + v7) + *v13;
            if ( *v15 > v16 )
            {
              *v15 = v16;
              v17 = v24;
              v18 = 72 * v24;
              *(_QWORD *)(*(_QWORD *)a1 + v18 + 8) = v6;
              *(_QWORD *)(*(_QWORD *)a1 + v18 + 16) = v10;
              v19 = *(_QWORD *)a1 + v18;
              if ( !*(_BYTE *)(v19 + 24) )
              {
                v20 = v30;
                if ( v30 == (__int64 *)(v32 - 8) )
                {
                  v23 = v7;
                  sub_FE0450(v25, &v24);
                  v7 = v23;
                  v19 = *(_QWORD *)a1 + 72 * v24;
                }
                else
                {
                  if ( v30 )
                  {
                    *v30 = v17;
                    v20 = v30;
                    v19 = *(_QWORD *)a1 + 72 * v24;
                  }
                  v30 = v20 + 1;
                }
                *(_BYTE *)(v19 + 24) = 1;
              }
              v9 = *(_QWORD *)(a1 + 24);
            }
          }
          ++v10;
          v12 = *(_QWORD *)(v9 + 24 * v6);
        }
        while ( v10 < 0x6DB6DB6DB6DB6DB7LL * ((*(_QWORD *)(v9 + 24 * v6 + 8) - v12) >> 3) );
      }
    }
  }
  v8 = *(_QWORD *)(*(_QWORD *)a1 + 72LL * *(_QWORD *)(a1 + 56));
LABEL_28:
  sub_2A5BF70((unsigned __int64 *)v25);
  return v8 != 0x4000000000000LL;
}
