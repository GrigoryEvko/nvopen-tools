// Function: sub_21E10C0
// Address: 0x21e10c0
//
__int64 __fastcall sub_21E10C0(__int64 a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  __int64 v7; // rsi
  _QWORD *v8; // r15
  __int64 v9; // rdx
  int v10; // eax
  __int64 v11; // rcx
  __int64 v12; // rdi
  _QWORD *v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // rdi
  int v16; // edx
  __int64 v17; // rsi
  __int64 v18; // rdi
  __int64 v19; // r9
  int v20; // edx
  __int64 v21; // rcx
  int v22; // r8d
  __int64 v23; // r12
  int v25; // [rsp+0h] [rbp-90h]
  __int64 v26; // [rsp+8h] [rbp-88h]
  __int64 v27; // [rsp+10h] [rbp-80h] BYREF
  int v28; // [rsp+18h] [rbp-78h]
  __int64 v29; // [rsp+20h] [rbp-70h] BYREF
  int v30; // [rsp+28h] [rbp-68h]
  __int64 v31; // [rsp+30h] [rbp-60h] BYREF
  int v32; // [rsp+38h] [rbp-58h]
  __int64 v33; // [rsp+40h] [rbp-50h]
  int v34; // [rsp+48h] [rbp-48h]
  __m128i v35; // [rsp+50h] [rbp-40h]

  v7 = *(_QWORD *)(a2 + 72);
  v8 = *(_QWORD **)(a1 - 176);
  v27 = v7;
  if ( v7 )
  {
    sub_1623A60((__int64)&v27, v7, 2);
    v7 = *(_QWORD *)(a2 + 72);
  }
  v9 = *(_QWORD *)(a2 + 32);
  v10 = *(_DWORD *)(a2 + 64);
  v11 = *(_QWORD *)(v9 + 80);
  v28 = v10;
  v12 = *(_QWORD *)(v11 + 88);
  v13 = *(_QWORD **)(v12 + 24);
  if ( *(_DWORD *)(v12 + 32) > 0x40u )
    v13 = (_QWORD *)*v13;
  v14 = *(_QWORD *)(*(_QWORD *)(v9 + 120) + 88LL);
  if ( *(_DWORD *)(v14 + 32) <= 0x40u )
    v26 = *(_QWORD *)(v14 + 24);
  else
    v26 = **(_QWORD **)(v14 + 24);
  v29 = v7;
  if ( v7 )
  {
    v25 = (int)v13;
    sub_1623A60((__int64)&v29, v7, 2);
    v10 = *(_DWORD *)(a2 + 64);
    LODWORD(v13) = v25;
  }
  v15 = *(_QWORD *)(a1 - 176);
  v30 = v10;
  v31 = sub_1D38BB0(v15, (unsigned int)v13, (__int64)&v29, 5, 0, 1, a3, a4, a5, 0);
  v32 = v16;
  if ( v29 )
    sub_161E7C0((__int64)&v29, v29);
  v17 = *(_QWORD *)(a2 + 72);
  v29 = v17;
  if ( v17 )
    sub_1623A60((__int64)&v29, v17, 2);
  v18 = *(_QWORD *)(a1 - 176);
  v30 = *(_DWORD *)(a2 + 64);
  v33 = sub_1D38BB0(v18, (unsigned int)v26, (__int64)&v29, 5, 0, 1, a3, a4, a5, 0);
  v34 = v20;
  if ( v29 )
    sub_161E7C0((__int64)&v29, v29);
  v21 = *(_QWORD *)(a2 + 40);
  v22 = *(_DWORD *)(a2 + 60);
  v35 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 32));
  v23 = sub_1D23DE0(v8, 197, (__int64)&v27, v21, v22, v19, &v31, 3);
  if ( v27 )
    sub_161E7C0((__int64)&v27, v27);
  return v23;
}
