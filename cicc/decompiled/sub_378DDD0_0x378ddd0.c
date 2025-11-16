// Function: sub_378DDD0
// Address: 0x378ddd0
//
void __fastcall sub_378DDD0(
        __int64 *a1,
        unsigned __int64 a2,
        unsigned __int64 a3,
        int a4,
        __m128i a5,
        __int64 a6,
        unsigned int a7)
{
  __int64 v8; // r15
  __int64 v9; // rsi
  __int16 v10; // r9d^2
  __int64 v11; // r8
  __int128 v12; // rax
  __int64 v13; // r9
  unsigned __int8 *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r8
  __int128 v19; // [rsp-20h] [rbp-B0h]
  __int16 v21; // [rsp+Ah] [rbp-86h]
  __int64 v22; // [rsp+10h] [rbp-80h]
  __int16 v23; // [rsp+12h] [rbp-7Eh]
  __int64 v24; // [rsp+18h] [rbp-78h]
  unsigned __int16 v26; // [rsp+26h] [rbp-6Ah]
  _QWORD *v27; // [rsp+28h] [rbp-68h]
  __int64 v28; // [rsp+28h] [rbp-68h]
  __int16 v29; // [rsp+2Ah] [rbp-66h]
  unsigned __int64 v30; // [rsp+38h] [rbp-58h]
  __int64 v31; // [rsp+40h] [rbp-50h] BYREF
  int v32; // [rsp+48h] [rbp-48h]

  if ( *(_DWORD *)(a2 + 68) )
  {
    v24 = *(unsigned int *)(a2 + 68);
    v8 = 0;
    do
    {
      while ( a4 == (_DWORD)v8 )
      {
LABEL_7:
        if ( ++v8 == v24 )
          return;
      }
      v16 = *(_QWORD *)(a2 + 48) + 16 * v8;
      LOWORD(a7) = *(_WORD *)v16;
      v26 = *(_WORD *)v16;
      v28 = *(_QWORD *)(v16 + 8);
      sub_2FE6CC0((__int64)&v31, *a1, *(_QWORD *)(a1[1] + 64), a7, v28);
      v11 = v28;
      if ( (_BYTE)v31 != 7 )
      {
        v9 = *(_QWORD *)(a2 + 80);
        v31 = v9;
        if ( v9 )
        {
          v23 = v10;
          sub_B96E90((__int64)&v31, v9, 1);
          v10 = v23;
          v11 = v28;
        }
        v21 = v10;
        v22 = v11;
        v27 = (_QWORD *)a1[1];
        v32 = *(_DWORD *)(a2 + 72);
        *(_QWORD *)&v12 = sub_3400EE0((__int64)v27, 0, (__int64)&v31, 0, a5);
        v30 = (unsigned int)v8 | v30 & 0xFFFFFFFF00000000LL;
        *((_QWORD *)&v19 + 1) = v30;
        *(_QWORD *)&v19 = a3;
        v14 = sub_3406EB0(v27, 0xA1u, (__int64)&v31, v26, v22, v13, v19, v12);
        sub_3760E70((__int64)a1, a2, (unsigned int)v8, (unsigned __int64)v14, v15);
        HIWORD(a7) = v21;
        if ( v31 )
        {
          sub_B91220((__int64)&v31, v31);
          HIWORD(a7) = v21;
        }
        goto LABEL_7;
      }
      v17 = (unsigned int)v8;
      v18 = (unsigned int)v8;
      v29 = v10;
      ++v8;
      sub_3760B50((__int64)a1, a2, v17, a3, v18);
      HIWORD(a7) = v29;
    }
    while ( v8 != v24 );
  }
}
