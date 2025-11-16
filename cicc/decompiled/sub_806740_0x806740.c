// Function: sub_806740
// Address: 0x806740
//
__int64 __fastcall sub_806740(_QWORD *a1)
{
  __int64 v2; // r14
  __m128i *v3; // rdi
  bool v4; // zf
  __int64 *v5; // rax
  _QWORD *v6; // rcx
  __int64 v8; // rdi
  __int64 v9; // rdi
  __int64 v10; // [rsp+8h] [rbp-78h]
  int v11; // [rsp+14h] [rbp-6Ch]
  int v12; // [rsp+18h] [rbp-68h]
  __int16 v13; // [rsp+1Ch] [rbp-64h]
  unsigned __int16 v14; // [rsp+1Eh] [rbp-62h]
  __m128i *v15; // [rsp+20h] [rbp-60h] BYREF
  __m128i *v16; // [rsp+28h] [rbp-58h] BYREF
  __m128i v17[5]; // [rsp+30h] [rbp-50h] BYREF

  v2 = a1[4];
  v3 = (__m128i *)a1[10];
  v4 = v3[2].m128i_i8[8] == 19;
  v11 = dword_4D03F38[0];
  v10 = *(_QWORD *)(*(_QWORD *)(v2 + 40) + 32LL);
  v14 = dword_4D03F38[1];
  v12 = dword_4F07508[0];
  v13 = dword_4F07508[1];
  *(_QWORD *)dword_4D03F38 = v3->m128i_i64[0];
  *(_QWORD *)dword_4F07508 = *(_QWORD *)dword_4D03F38;
  if ( v4 )
  {
    sub_7E7150(v3, (__int64)v17, &v15);
    sub_7E91D0(a1[11], (__int64)v17);
    if ( (*(_BYTE *)(v10 + 176) & 0x10) != 0 )
    {
      v9 = a1[4];
      if ( (*(_BYTE *)(v9 + 205) & 0x1C) == 8 )
        sub_7FDF40(v9, 1, 1);
    }
    sub_7DE0F0((__int64)v15, 1u, 0);
  }
  else
  {
    v15 = (__m128i *)v3[4].m128i_i64[1];
    sub_7E1740((__int64)v3, (__int64)v17);
    sub_7E91D0(a1[11], (__int64)v17);
    if ( (*(_BYTE *)(v10 + 176) & 0x10) != 0 )
    {
      v8 = a1[4];
      if ( (*(_BYTE *)(v8 + 205) & 0x1C) == 8 )
        sub_7FDF40(v8, 1, 1);
    }
    sub_806570((__int64)a1, v17);
    sub_7EDD70(v15, &v16);
  }
  if ( (*(_BYTE *)(v2 + 194) & 0x20) == 0 )
  {
    v5 = (__int64 *)a1[6];
    if ( v5 )
    {
      v6 = 0;
      while ( !*((_BYTE *)v5 + 8) )
      {
        v6 = v5;
        if ( !*v5 )
          goto LABEL_12;
        v5 = (__int64 *)*v5;
      }
      if ( v6 )
        *v6 = 0;
      else
        a1[6] = 0;
    }
  }
LABEL_12:
  dword_4F07508[0] = v12;
  LOWORD(dword_4F07508[1]) = v13;
  dword_4D03F38[0] = v11;
  LOWORD(dword_4D03F38[1]) = v14;
  return v14;
}
