// Function: sub_8C27C0
// Address: 0x8c27c0
//
unsigned __int64 __fastcall sub_8C27C0(
        __int64 a1,
        int a2,
        __int64 a3,
        __int64 *a4,
        unsigned int a5,
        _DWORD *a6,
        __int64 a7)
{
  unsigned __int64 v7; // r12
  int v10; // r15d
  const __m128i *v12; // rax
  _DWORD *v13; // r11
  __int8 v14; // al
  __int64 **v15; // rsi
  __int64 **v16; // rax
  __int64 v17; // r8
  __int64 v18; // r9
  _QWORD *v19; // rax
  __int64 **v22; // [rsp+18h] [rbp-38h] BYREF

  v7 = *(_QWORD *)a1;
  if ( (*(_BYTE *)(*(_QWORD *)a1 + 81LL) & 0x10) != 0 && (*(_BYTE *)(*(_QWORD *)(v7 + 64) + 177LL) & 0x20) != 0 )
  {
    v10 = a3;
    v12 = sub_8A1CE0(**(_QWORD **)(a1 + 248), *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL), a2, a3, a4, 0, 0, a5, a6, a7);
    v13 = a6;
    v7 = (unsigned __int64)v12;
    if ( v12 )
    {
      v14 = v12[5].m128i_i8[0];
      if ( v14 == 16 )
      {
        v7 = **(_QWORD **)(v7 + 88);
        v14 = *(_BYTE *)(v7 + 80);
      }
      if ( v14 == 24 )
      {
        v7 = *(_QWORD *)(v7 + 88);
        if ( !v7 )
          goto LABEL_11;
        v14 = *(_BYTE *)(v7 + 80);
      }
      if ( v14 == 20 )
      {
        v15 = *(__int64 ***)(a1 + 240);
        v22 = v15;
        if ( !v15 )
          return v7;
        v16 = (__int64 **)sub_690FF0(
                            v7,
                            (int)v15,
                            **(_QWORD **)(*(_QWORD *)(v7 + 88) + 328LL),
                            a2,
                            v10,
                            (__int64)a4,
                            a5,
                            (__int64)a6,
                            a7);
        v13 = a6;
        v22 = v16;
        if ( !*a6 )
        {
          v19 = sub_8B74F0(v7, &v22, 1u, a4, v17, v18);
          v13 = a6;
          v7 = (unsigned __int64)v19;
          if ( v19 )
            return v7;
        }
      }
      else if ( (unsigned __int8)(v14 - 10) <= 1u || v14 == 17 )
      {
        return v7;
      }
    }
LABEL_11:
    *v13 = 1;
    return 0;
  }
  return v7;
}
