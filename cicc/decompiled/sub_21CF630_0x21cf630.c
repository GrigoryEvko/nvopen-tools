// Function: sub_21CF630
// Address: 0x21cf630
//
__int64 __fastcall sub_21CF630(double a1, double a2, double a3, __int64 a4, __int64 a5, __int64 a6, __int64 *a7)
{
  __int64 v8; // rcx
  __int64 v10; // rsi
  unsigned int v11; // ebx
  __int64 v12; // rax
  unsigned int v13; // eax
  unsigned __int8 v14; // bl
  unsigned int v15; // edx
  __int64 v16; // r12
  __int128 v18; // [rsp-20h] [rbp-80h]
  __int64 v19; // [rsp+8h] [rbp-58h]
  __int64 v20; // [rsp+8h] [rbp-58h]
  __int64 *v21; // [rsp+10h] [rbp-50h]
  __int64 v22; // [rsp+20h] [rbp-40h] BYREF
  int v23; // [rsp+28h] [rbp-38h]

  v8 = a5;
  v10 = *(_QWORD *)(a5 + 72);
  v22 = v10;
  if ( v10 )
  {
    v19 = v8;
    sub_1623A60((__int64)&v22, v10, 2);
    v8 = v19;
  }
  v20 = v8;
  v23 = *(_DWORD *)(v8 + 64);
  v11 = sub_1D19C00(v8);
  v12 = sub_1E0A0C0(a7[4]);
  v13 = 8 * sub_15A9520(v12, v11);
  if ( v13 == 32 )
  {
    v14 = 5;
  }
  else if ( v13 > 0x20 )
  {
    v14 = 6;
    if ( v13 != 64 )
    {
      v14 = 0;
      if ( v13 == 128 )
        v14 = 7;
    }
  }
  else
  {
    v14 = 3;
    if ( v13 != 8 )
      v14 = 4 * (v13 == 16);
  }
  v21 = sub_1D29600(a7, *(_QWORD *)(v20 + 88), (__int64)&v22, v14, 0, 0, 1, 0);
  *((_QWORD *)&v18 + 1) = v15 | a6 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v18 = v21;
  v16 = sub_1D309E0(a7, 260, (__int64)&v22, v14, 0, 0, a1, a2, a3, v18);
  if ( v22 )
    sub_161E7C0((__int64)&v22, v22);
  return v16;
}
