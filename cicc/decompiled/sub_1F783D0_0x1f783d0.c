// Function: sub_1F783D0
// Address: 0x1f783d0
//
__int64 *__fastcall sub_1F783D0(__int64 **a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 *v6; // rax
  __int64 v7; // r14
  unsigned __int64 v8; // r15
  __int64 v9; // r10
  __int64 v10; // r11
  int v11; // edx
  __int16 v12; // ax
  __int64 *v13; // r14
  __int64 *v14; // rax
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 *v18; // rbx
  unsigned __int64 v19; // r9
  __int64 v20; // rcx
  const void **v21; // r8
  __int128 v22; // [rsp-10h] [rbp-80h]
  const void **v23; // [rsp+8h] [rbp-68h]
  __int64 v24; // [rsp+10h] [rbp-60h]
  __int64 v25; // [rsp+18h] [rbp-58h]
  __int64 v26; // [rsp+20h] [rbp-50h]
  unsigned int v27; // [rsp+28h] [rbp-48h]
  __int64 v28; // [rsp+30h] [rbp-40h] BYREF
  int v29; // [rsp+38h] [rbp-38h]

  v6 = *(__int64 **)(a2 + 32);
  v7 = *v6;
  v8 = v6[1];
  v9 = v6[5];
  v10 = v6[6];
  v11 = *(unsigned __int16 *)(*v6 + 24);
  v12 = *(_WORD *)(v9 + 24);
  if ( (v11 == 11 || v11 == 33) && (v12 == 11 || v12 == 33) )
  {
    v16 = *(_QWORD *)(a2 + 40);
    v17 = *(_QWORD *)(a2 + 72);
    v18 = *a1;
    v19 = *(unsigned __int16 *)(a2 + 80);
    v20 = **(unsigned __int8 **)(a2 + 40);
    v21 = *(const void ***)(v16 + 8);
    v28 = v17;
    if ( v17 )
    {
      v23 = v21;
      v26 = v20;
      v24 = v9;
      v25 = v10;
      v27 = v19;
      sub_1623A60((__int64)&v28, v17, 2);
      v21 = v23;
      v20 = v26;
      v9 = v24;
      v10 = v25;
      v19 = v27;
    }
    *((_QWORD *)&v22 + 1) = v10;
    *(_QWORD *)&v22 = v9;
    v29 = *(_DWORD *)(a2 + 64);
    v13 = sub_1D332F0(v18, 80, (__int64)&v28, v20, v21, v19, a3, a4, a5, v7, v8, v22);
    if ( v28 )
      sub_161E7C0((__int64)&v28, v28);
  }
  else
  {
    v13 = 0;
    v14 = sub_1F77C50(a1, a2, a3, a4, a5);
    if ( v14 )
      return v14;
  }
  return v13;
}
