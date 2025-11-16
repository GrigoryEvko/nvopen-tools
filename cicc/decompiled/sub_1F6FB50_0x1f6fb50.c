// Function: sub_1F6FB50
// Address: 0x1f6fb50
//
__int64 __fastcall sub_1F6FB50(__int64 a1, _QWORD *a2, __m128i a3, double a4, __m128i a5)
{
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // r13
  int v9; // r8d
  __int64 v10; // rcx
  int v11; // r15d
  char *v12; // rax
  unsigned __int8 v13; // dl
  const void **v14; // r14
  int v15; // eax
  int v16; // esi
  __int64 v17; // r9
  __int64 result; // rax
  _QWORD *v19; // r12
  __int64 v20; // [rsp+8h] [rbp-78h]
  int v21; // [rsp+14h] [rbp-6Ch]
  unsigned __int8 v22; // [rsp+18h] [rbp-68h]
  __int64 v23; // [rsp+18h] [rbp-68h]
  __int64 v24; // [rsp+20h] [rbp-60h] BYREF
  int v25; // [rsp+28h] [rbp-58h]
  __int64 v26; // [rsp+30h] [rbp-50h] BYREF
  int v27; // [rsp+38h] [rbp-48h]
  __int64 v28; // [rsp+40h] [rbp-40h]
  int v29; // [rsp+48h] [rbp-38h]

  v6 = *(_QWORD *)(a1 + 32);
  v7 = *(_QWORD *)(a1 + 72);
  v8 = *(_QWORD *)v6;
  v9 = *(_DWORD *)(v6 + 8);
  v10 = *(_QWORD *)(v6 + 40);
  v11 = *(_DWORD *)(v6 + 48);
  v12 = *(char **)(a1 + 40);
  v13 = *v12;
  v14 = (const void **)*((_QWORD *)v12 + 1);
  v24 = v7;
  v22 = v13;
  if ( v7 )
  {
    v20 = v10;
    v21 = v9;
    sub_1623A60((__int64)&v24, v7, 2);
    v10 = v20;
    v9 = v21;
  }
  v15 = *(_DWORD *)(a1 + 64);
  v16 = *(unsigned __int16 *)(a1 + 24);
  v29 = v11;
  v28 = v10;
  v25 = v15;
  v26 = v8;
  v27 = v9;
  if ( sub_1D18610((__int64)a2, v16, (__int64)&v26) )
  {
    v26 = 0;
    v27 = 0;
    v19 = sub_1D2B300(a2, 0x30u, (__int64)&v26, v22, (__int64)v14, v17);
    if ( v26 )
      sub_161E7C0((__int64)&v26, v26);
    result = (__int64)v19;
  }
  else
  {
    result = 0;
    if ( *(_WORD *)(v8 + 24) == 48 )
      result = sub_1D38BB0((__int64)a2, 0, (__int64)&v24, v22, v14, 0, a3, a4, a5, 0);
  }
  if ( v24 )
  {
    v23 = result;
    sub_161E7C0((__int64)&v24, v24);
    return v23;
  }
  return result;
}
