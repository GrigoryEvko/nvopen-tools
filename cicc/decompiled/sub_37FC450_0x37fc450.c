// Function: sub_37FC450
// Address: 0x37fc450
//
unsigned __int8 *__fastcall sub_37FC450(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v4; // r12
  _QWORD *v5; // r14
  unsigned __int8 *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx
  unsigned __int16 v9; // ax
  __int64 v10; // rdx
  __int64 v11; // rax
  char v12; // cl
  __int64 v13; // rax
  int v14; // edx
  __int64 v15; // r9
  unsigned __int16 v16; // ax
  __int64 v17; // r13
  __int64 v18; // rdx
  __int64 v19; // rsi
  __int64 v20; // rcx
  unsigned __int8 *v21; // r12
  __int64 v23; // [rsp+8h] [rbp-68h]
  __int128 v24; // [rsp+10h] [rbp-60h]
  unsigned __int16 v25; // [rsp+20h] [rbp-50h] BYREF
  __int64 v26; // [rsp+28h] [rbp-48h]
  __int64 v27; // [rsp+30h] [rbp-40h] BYREF
  int v28; // [rsp+38h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 40);
  v5 = *(_QWORD **)(a1 + 8);
  v6 = sub_33FB890((__int64)v5, 9u, 0, *(_QWORD *)v4, *(_QWORD *)(v4 + 8), a3);
  *((_QWORD *)&v24 + 1) = v7;
  v8 = *(_QWORD *)(a2 + 48);
  *(_QWORD *)&v24 = v6;
  v9 = *(_WORD *)v8;
  v10 = *(_QWORD *)(v8 + 8);
  v25 = v9;
  v26 = v10;
  if ( v9 )
  {
    if ( (unsigned __int16)(v9 - 17) <= 0xD3u )
    {
      LOWORD(v27) = v9;
      v17 = 0;
      v16 = sub_30369B0((unsigned __int16 *)&v27);
    }
    else
    {
      if ( v9 == 1 || (unsigned __int16)(v9 - 504) <= 7u )
        BUG();
      v11 = 16LL * (v9 - 1);
      v12 = byte_444C4A0[v11 + 8];
      v13 = *(_QWORD *)&byte_444C4A0[v11];
      LOBYTE(v28) = v12;
      v27 = v13;
      v14 = sub_CA1930(&v27);
      v16 = 2;
      if ( v14 != 1 )
      {
        v16 = 3;
        if ( v14 != 2 )
        {
          v16 = 4;
          if ( v14 != 4 )
          {
            v16 = 5;
            if ( v14 != 8 )
            {
              v16 = 6;
              if ( v14 != 16 )
              {
                v16 = 7;
                if ( v14 != 32 )
                {
                  v16 = 8;
                  if ( v14 != 64 )
                    v16 = 9 * (v14 == 128);
                }
              }
            }
          }
        }
      }
      v17 = 0;
    }
  }
  else
  {
    if ( sub_30070B0((__int64)&v25) )
      v16 = sub_300A990(&v25, 9);
    else
      v16 = sub_30072B0((__int64)&v25);
    v17 = v18;
  }
  v19 = *(_QWORD *)(a2 + 80);
  v20 = v16;
  v27 = v19;
  if ( v19 )
  {
    v23 = v16;
    sub_B96E90((__int64)&v27, v19, 1);
    v20 = v23;
  }
  v28 = *(_DWORD *)(a2 + 72);
  v21 = sub_3406EB0(v5, 0x35u, (__int64)&v27, v20, v17, v15, v24, *(_OWORD *)(v4 + 40));
  if ( v27 )
    sub_B91220((__int64)&v27, v27);
  return v21;
}
