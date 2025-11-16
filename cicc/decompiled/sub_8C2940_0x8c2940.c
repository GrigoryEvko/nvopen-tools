// Function: sub_8C2940
// Address: 0x8c2940
//
__int64 *__fastcall sub_8C2940(__int64 a1, int a2, __int64 a3, __int64 *a4, unsigned int a5, _DWORD *a6, __int64 a7)
{
  int v7; // r10d
  int v9; // r14d
  __int64 v12; // rax
  __m128i *v13; // rsi
  __int64 *v14; // rax
  __int64 v15; // r12
  char v16; // al
  __int64 **v17; // rdx
  __int64 *v18; // rax
  __int64 v19; // rdx
  __int64 *v20; // r12
  const __m128i *v22; // rax
  __int8 v23; // al
  __int64 v24; // rax
  _QWORD *v25; // rbx
  _QWORD *v26; // rax
  __m128i *v28; // [rsp+18h] [rbp-38h] BYREF

  v7 = a5;
  v9 = a3;
  v12 = *(_QWORD *)(a1 + 216);
  v13 = *(__m128i **)v12;
  v14 = *(__int64 **)(v12 + 16);
  v28 = v13;
  v15 = *v14;
  if ( (*(_BYTE *)(*v14 + 81) & 0x10) != 0 && (*(_BYTE *)(*(_QWORD *)(v15 + 64) + 177LL) & 0x20) != 0 )
  {
    v22 = sub_8A1CE0(v15, *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL), a2, a3, a4, 0, 0, a5, a6, a7);
    v20 = (__int64 *)v22;
    if ( v22 )
    {
      v23 = v22[5].m128i_i8[0];
      v7 = a5;
      if ( v23 == 16 )
      {
        v20 = *(__int64 **)v20[11];
        v23 = *((_BYTE *)v20 + 80);
      }
      if ( v23 == 24 )
      {
        v20 = (__int64 *)v20[11];
        if ( !v20 )
          goto LABEL_8;
        v23 = *((_BYTE *)v20 + 80);
      }
      switch ( v23 )
      {
        case 9:
          return v20;
        case 21:
          LODWORD(v13) = (_DWORD)v28;
          v15 = **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v20[11] + 192) + 216LL) + 16LL);
          goto LABEL_3;
        case 2:
          v24 = v20[11];
          if ( v24 )
          {
            if ( *(_BYTE *)(v24 + 173) == 12 )
            {
              if ( v28 )
              {
                v25 = sub_724D80(12);
                sub_7249B0((__int64)v25, 11);
                v25[23] = v20[11];
                v25[24] = v28;
                v25[16] = dword_4D03B80;
                v26 = sub_87EBB0(2u, *v20, v20 + 6);
                v26[11] = v25;
                return v26;
              }
              return v20;
            }
          }
          break;
      }
    }
LABEL_8:
    *a6 = 1;
    return 0;
  }
LABEL_3:
  v16 = *(_BYTE *)(v15 + 80);
  v17 = *(__int64 ***)(v15 + 88);
  if ( v16 == 20 )
  {
    v19 = *v17[41];
  }
  else
  {
    if ( v16 == 21 )
      v18 = v17[29];
    else
      v18 = v17[4];
    v19 = *v18;
  }
  v28 = (__m128i *)sub_690FF0(v15, (int)v13, v19, a2, v9, (__int64)a4, v7, (__int64)a6, a7);
  if ( *a6 )
    goto LABEL_8;
  v20 = sub_8C0230(v15, &v28, 0, 1u, 0);
  if ( !v20 )
    goto LABEL_8;
  return v20;
}
