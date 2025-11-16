// Function: sub_1F705B0
// Address: 0x1f705b0
//
__int64 __fastcall sub_1F705B0(__int64 **a1, __int64 a2, double a3, double a4, double a5)
{
  __int64 *v6; // rax
  __int64 v7; // r13
  __int64 v8; // r10
  __int64 v9; // r11
  const void **v10; // r12
  unsigned int v11; // r15d
  int v12; // eax
  char v13; // al
  __int64 v14; // rsi
  __int64 *v15; // r13
  __int64 v16; // r13
  __int16 v18; // ax
  __int128 v19; // [rsp-10h] [rbp-60h]
  __int64 v20; // [rsp+0h] [rbp-50h]
  __int64 v21; // [rsp+0h] [rbp-50h]
  __int64 v22; // [rsp+8h] [rbp-48h]
  __int64 v23; // [rsp+8h] [rbp-48h]
  __int64 v24; // [rsp+10h] [rbp-40h] BYREF
  int v25; // [rsp+18h] [rbp-38h]

  v6 = *(__int64 **)(a2 + 32);
  v7 = *v6;
  v8 = *v6;
  v9 = v6[1];
  v10 = *(const void ***)(*(_QWORD *)(a2 + 40) + 8LL);
  v11 = **(unsigned __int8 **)(a2 + 40);
  v12 = *(unsigned __int16 *)(*v6 + 24);
  if ( v12 != 11 && v12 != 33 )
  {
    v20 = v8;
    v22 = v9;
    v13 = sub_1D16930(v7);
    v8 = v20;
    v9 = v22;
    if ( !v13 )
    {
      v18 = *(_WORD *)(v7 + 24);
      if ( v18 > 177 )
      {
        if ( v18 != 179 )
          return 0;
      }
      else if ( v18 <= 173 )
      {
        return 0;
      }
      return v20;
    }
  }
  v14 = *(_QWORD *)(a2 + 72);
  v15 = *a1;
  v24 = v14;
  if ( v14 )
  {
    v21 = v8;
    v23 = v9;
    sub_1623A60((__int64)&v24, v14, 2);
    v8 = v21;
    v9 = v23;
  }
  *((_QWORD *)&v19 + 1) = v9;
  *(_QWORD *)&v19 = v8;
  v25 = *(_DWORD *)(a2 + 64);
  v16 = sub_1D309E0(v15, 175, (__int64)&v24, v11, v10, 0, a3, a4, a5, v19);
  if ( v24 )
    sub_161E7C0((__int64)&v24, v24);
  return v16;
}
