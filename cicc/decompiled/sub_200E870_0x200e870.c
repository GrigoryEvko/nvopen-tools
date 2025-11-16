// Function: sub_200E870
// Address: 0x200e870
//
unsigned __int64 __fastcall sub_200E870(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        _QWORD *a5,
        __m128i a6,
        double a7,
        __m128i a8)
{
  __int64 v8; // r15
  __int64 v13; // rax
  char v14; // di
  __int64 v15; // rax
  unsigned int v16; // esi
  __int64 v17; // r9
  bool v18; // cc
  __int64 v19; // rax
  const void **v20; // r8
  const void **v22; // rdx
  unsigned int v23; // eax
  __int64 v24; // [rsp+8h] [rbp-48h]
  char v26[8]; // [rsp+10h] [rbp-40h] BYREF
  __int64 v27; // [rsp+18h] [rbp-38h]

  v13 = *(_QWORD *)(a2 + 40) + 16LL * (unsigned int)a3;
  v14 = *(_BYTE *)v13;
  v15 = *(_QWORD *)(v13 + 8);
  v26[0] = v14;
  v27 = v15;
  if ( v14 )
  {
    v16 = (unsigned int)sub_200D0E0(v14) >> 1;
    v18 = v16 <= 0x20;
    if ( v16 != 32 )
      goto LABEL_3;
LABEL_14:
    LOBYTE(v19) = 5;
    goto LABEL_6;
  }
  v23 = sub_1F58D40((__int64)v26);
  v17 = a4;
  v16 = v23 >> 1;
  v18 = v23 >> 1 <= 0x20;
  if ( v23 >> 1 == 32 )
    goto LABEL_14;
LABEL_3:
  if ( v18 )
  {
    if ( v16 == 8 )
    {
      LOBYTE(v19) = 3;
    }
    else
    {
      LOBYTE(v19) = 4;
      if ( v16 != 16 )
      {
        LOBYTE(v19) = 2;
        if ( v16 != 1 )
          goto LABEL_12;
      }
    }
  }
  else if ( v16 == 64 )
  {
    LOBYTE(v19) = 6;
  }
  else
  {
    if ( v16 != 128 )
    {
LABEL_12:
      v24 = v17;
      v19 = sub_1F58CC0(*(_QWORD **)(*(_QWORD *)(a1 + 8) + 48LL), v16);
      v17 = v24;
      v8 = v19;
      v20 = v22;
      goto LABEL_7;
    }
    LOBYTE(v19) = 7;
  }
LABEL_6:
  v20 = 0;
LABEL_7:
  LOBYTE(v8) = v19;
  return sub_200E3C0((__int64 **)a1, a2, a3, v8, v20, v17, a6, a7, a8, v8, v20, a5);
}
