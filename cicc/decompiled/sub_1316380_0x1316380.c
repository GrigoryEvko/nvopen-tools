// Function: sub_1316380
// Address: 0x1316380
//
__int64 __fastcall sub_1316380(__int64 a1, __int64 a2, unsigned __int64 a3, unsigned __int64 a4, unsigned __int8 a5)
{
  int v9; // r9d
  __int64 v10; // rax
  unsigned __int8 v11; // al
  __int64 result; // rax
  __int64 v13; // rsi
  char v14; // cl
  __int64 v15; // rdx
  unsigned int v16; // edx
  __int64 v17; // rdx
  unsigned __int64 v18; // rdx
  __int64 (__fastcall **v19)(int, int, int, int, int, int, int); // rax
  unsigned __int64 v20; // rax
  char v21; // cl
  __int64 v22; // rax
  unsigned int v23; // [rsp+4h] [rbp-4Ch]
  unsigned __int64 v24; // [rsp+8h] [rbp-48h]
  char v25; // [rsp+17h] [rbp-39h] BYREF
  char v26; // [rsp+18h] [rbp-38h] BYREF

  v25 = 0;
  if ( a3 > 0x1000 )
  {
    if ( a3 > 0x7000000000000000LL )
    {
      v9 = 232;
    }
    else
    {
      v21 = 7;
      _BitScanReverse64((unsigned __int64 *)&v22, 2 * a3 - 1);
      if ( (unsigned int)v22 >= 7 )
        v21 = v22;
      if ( (unsigned int)v22 < 6 )
        LODWORD(v22) = 6;
      v9 = ((((a3 - 1) & (-1LL << (v21 - 3))) >> (v21 - 3)) & 3) + 4 * v22 - 23;
    }
  }
  else
  {
    v9 = byte_5060800[(a3 + 7) >> 3];
  }
  v23 = v9;
  v24 = a3 + *(_QWORD *)&dword_50607C0;
  v10 = sub_1316370(a2);
  if ( !qword_4F969E8 )
    goto LABEL_4;
  v19 = *(__int64 (__fastcall ***)(int, int, int, int, int, int, int))(v10 + 8);
  if ( !a1 || v19 != &off_49E8020 )
    goto LABEL_4;
  v20 = *(_QWORD *)(a1 + 128);
  if ( v20 > 1 )
  {
    *(_QWORD *)(a1 + 128) = v20 - 1;
LABEL_4:
    v11 = 0;
    goto LABEL_5;
  }
  if ( a4 > 0x1000 || (v20 & 1) == 0 || v24 + 0x2000 > 0x7000000000000000LL )
    goto LABEL_4;
  *(_QWORD *)(a1 + 128) = qword_4F969E8;
  v11 = 1;
LABEL_5:
  result = sub_130B510(a1, a2 + 10648, v24, a4, 0, v23, a5, v11, (__int64)&v25);
  if ( result )
  {
    if ( a3 > 0x7000000000000000LL )
    {
      v17 = 10384;
    }
    else
    {
      v13 = 0x4000;
      v14 = 7;
      if ( a3 >= 0x4000 )
        v13 = a3;
      _BitScanReverse64((unsigned __int64 *)&v15, 2 * v13 - 1);
      if ( (unsigned int)v15 >= 7 )
        v14 = v15;
      if ( (unsigned int)v15 < 6 )
        LODWORD(v15) = 6;
      v16 = ((((v13 - 1) & (unsigned __int64)(-1LL << (v14 - 3))) >> (v14 - 3)) & 3) + 4 * v15 - 23;
      if ( v16 < 0x24 )
        v16 = 36;
      v17 = 48LL * (v16 - 36) + 976;
    }
    _InterlockedAdd64((volatile signed __int64 *)(a2 + v17), 1u);
    if ( *(_QWORD *)&dword_50607C0 && a4 <= 0xFFF )
    {
      _BitScanReverse64(&a4, (a4 + 63) & 0xFFFFFFFFFFFFFFC0LL);
      if ( a1 )
      {
        v18 = 0x5851F42D4C957F2DLL * *(_QWORD *)(a1 + 112) + 0x14057B7EF767814FLL;
        *(_QWORD *)(a1 + 112) = v18;
      }
      else
      {
        v18 = 0x5851F42D4C957F2DLL * (_QWORD)&v26 + 0x14057B7EF767814FLL;
      }
      *(_QWORD *)(result + 8) += v18 >> ((unsigned __int8)a4 + 52) << a4;
    }
  }
  return result;
}
