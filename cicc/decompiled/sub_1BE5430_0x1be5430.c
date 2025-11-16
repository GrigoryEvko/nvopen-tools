// Function: sub_1BE5430
// Address: 0x1be5430
//
__int64 __fastcall sub_1BE5430(__int64 a1, __int64 a2, unsigned int a3, double a4, double a5, double a6)
{
  __int64 *v9; // r15
  bool v10; // cc
  __int64 *v11; // rax
  __int64 v13; // rsi
  __int64 v14; // rdi
  __int64 v15; // r8
  __int64 *v16; // r9
  _QWORD *v17; // r14
  __int64 v18; // rsi
  unsigned int v19; // ecx
  __int64 v20; // rdx
  __int64 v22; // rax
  __int64 v23; // rdi
  unsigned __int64 v24; // rsi
  __int64 v25; // rax
  __int64 v26; // rsi
  __int64 v27; // rsi
  __int64 v28; // rdx
  unsigned __int8 *v29; // rsi
  __int64 v30; // r14
  __int64 v31; // rax
  __int64 v32; // rax
  unsigned __int64 *v33; // [rsp+8h] [rbp-88h]
  unsigned __int8 *v34; // [rsp+18h] [rbp-78h] BYREF
  __int64 v35; // [rsp+20h] [rbp-70h] BYREF
  __int16 v36; // [rsp+30h] [rbp-60h]
  __int64 v37[2]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v38; // [rsp+50h] [rbp-40h]

  v9 = *(__int64 **)(a2 + 176);
  v10 = (unsigned int)*(unsigned __int8 *)(a1 + 112) - 11 <= 0x11;
  v11 = *(__int64 **)(a1 + 80);
  v13 = *v11;
  if ( v10 )
  {
    v30 = sub_1BA16F0(a2, v13, a3);
    v31 = sub_1BA16F0(a2, *(_QWORD *)(*(_QWORD *)(a1 + 80) + 8LL), a3);
    v38 = 257;
    v32 = sub_1904E90((__int64)v9, *(unsigned __int8 *)(a1 + 112), v30, v31, v37, 0, a4, a5, a6);
    v18 = a1 + 40;
    v19 = a3;
    v20 = v32;
  }
  else
  {
    v36 = 257;
    v14 = sub_1BA16F0(a2, v13, a3);
    if ( *(_BYTE *)(v14 + 16) > 0x10u )
    {
      v38 = 257;
      v22 = sub_15FB630((__int64 *)v14, (__int64)v37, 0);
      v23 = v9[1];
      v17 = (_QWORD *)v22;
      if ( v23 )
      {
        v33 = (unsigned __int64 *)v9[2];
        sub_157E9D0(v23 + 40, v22);
        v24 = *v33;
        v25 = v17[3] & 7LL;
        v17[4] = v33;
        v24 &= 0xFFFFFFFFFFFFFFF8LL;
        v17[3] = v24 | v25;
        *(_QWORD *)(v24 + 8) = v17 + 3;
        *v33 = *v33 & 7 | (unsigned __int64)(v17 + 3);
      }
      sub_164B780((__int64)v17, &v35);
      v26 = *v9;
      if ( *v9 )
      {
        v34 = (unsigned __int8 *)*v9;
        sub_1623A60((__int64)&v34, v26, 2);
        v27 = v17[6];
        v28 = (__int64)(v17 + 6);
        if ( v27 )
        {
          sub_161E7C0((__int64)(v17 + 6), v27);
          v28 = (__int64)(v17 + 6);
        }
        v29 = v34;
        v17[6] = v34;
        if ( v29 )
          sub_1623210((__int64)&v34, v29, v28);
      }
    }
    else
    {
      v17 = (_QWORD *)sub_15A2B00((__int64 *)v14, a4, a5, a6);
    }
    v18 = a1 + 40;
    v19 = a3;
    v20 = (__int64)v17;
  }
  return sub_1BE4D70(a2, v18, v20, v19, v15, v16);
}
