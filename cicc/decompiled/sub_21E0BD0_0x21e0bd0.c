// Function: sub_21E0BD0
// Address: 0x21e0bd0
//
__int64 __fastcall sub_21E0BD0(__int64 a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  __int64 *v6; // rax
  __int64 v7; // rsi
  _QWORD *v8; // r12
  __int64 v9; // r14
  int v10; // r13d
  __int64 v11; // rdi
  __int64 v12; // rax
  _QWORD *v13; // rsi
  __int64 v14; // rax
  int v15; // edx
  __int64 v16; // rcx
  _BYTE *v17; // r9
  __int64 result; // rax
  __int64 v19; // rsi
  int v20; // r8d
  int v21; // [rsp+0h] [rbp-70h]
  int v22; // [rsp+0h] [rbp-70h]
  __int64 v23; // [rsp+8h] [rbp-68h]
  _BYTE *v24; // [rsp+8h] [rbp-68h]
  __int64 v25; // [rsp+8h] [rbp-68h]
  __int64 v26; // [rsp+10h] [rbp-60h] BYREF
  int v27; // [rsp+18h] [rbp-58h]
  __int64 v28; // [rsp+20h] [rbp-50h] BYREF
  int v29; // [rsp+28h] [rbp-48h]
  __int64 v30; // [rsp+30h] [rbp-40h]
  int v31; // [rsp+38h] [rbp-38h]

  v6 = *(__int64 **)(a2 + 32);
  v7 = *(_QWORD *)(a2 + 72);
  v8 = *(_QWORD **)(a1 - 176);
  v9 = *v6;
  v10 = *((_DWORD *)v6 + 2);
  v28 = v7;
  if ( v7 )
  {
    sub_1623A60((__int64)&v28, v7, 2);
    v6 = *(__int64 **)(a2 + 32);
    v11 = *(_QWORD *)(a1 - 176);
  }
  else
  {
    v11 = (__int64)v8;
  }
  v29 = *(_DWORD *)(a2 + 64);
  v12 = *(_QWORD *)(v6[10] + 88);
  v13 = *(_QWORD **)(v12 + 24);
  if ( *(_DWORD *)(v12 + 32) > 0x40u )
    v13 = (_QWORD *)*v13;
  v14 = sub_1D38BB0(v11, (unsigned int)v13, (__int64)&v28, 5, 0, 1, a3, a4, a5, 0);
  v16 = v14;
  if ( v28 )
  {
    v21 = v15;
    v23 = v14;
    sub_161E7C0((__int64)&v28, v28);
    v15 = v21;
    v16 = v23;
  }
  v17 = *(_BYTE **)(a2 + 40);
  result = 0;
  if ( *v17 == 5 )
  {
    v19 = *(_QWORD *)(a2 + 72);
    v30 = v9;
    v31 = v10;
    v20 = *(_DWORD *)(a2 + 60);
    v28 = v16;
    v29 = v15;
    v26 = v19;
    if ( v19 )
    {
      v22 = v20;
      v24 = v17;
      sub_1623A60((__int64)&v26, v19, 2);
      v20 = v22;
      v17 = v24;
    }
    v27 = *(_DWORD *)(a2 + 64);
    result = sub_1D23DE0(v8, 3264, (__int64)&v26, (__int64)v17, v20, (__int64)v17, &v28, 2);
    if ( v26 )
    {
      v25 = result;
      sub_161E7C0((__int64)&v26, v26);
      return v25;
    }
  }
  return result;
}
