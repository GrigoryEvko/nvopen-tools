// Function: sub_1F71080
// Address: 0x1f71080
//
__int64 *__fastcall sub_1F71080(__int64 **a1, __int64 a2, double a3, double a4, __m128i a5)
{
  _QWORD *v6; // rax
  __int64 v7; // r12
  __int64 v8; // r13
  int v9; // edx
  __int64 v10; // r14
  __int64 v11; // r15
  __int64 v12; // r10
  __int64 v13; // r11
  __int16 v14; // cx
  __int64 v15; // rcx
  __int64 *result; // rax
  __int64 v17; // rsi
  const void ***v18; // rcx
  int v19; // r8d
  __int64 v20; // rsi
  __int64 *v21; // r10
  const void ***v22; // rcx
  int v23; // r8d
  __int128 v24; // [rsp-30h] [rbp-A0h]
  __int128 v25; // [rsp-20h] [rbp-90h]
  __int128 v26; // [rsp-20h] [rbp-90h]
  __int128 v27; // [rsp-10h] [rbp-80h]
  __int128 v28; // [rsp-10h] [rbp-80h]
  const void ***v29; // [rsp+0h] [rbp-70h]
  int v30; // [rsp+8h] [rbp-68h]
  const void ***v31; // [rsp+8h] [rbp-68h]
  __int64 v32; // [rsp+10h] [rbp-60h]
  int v33; // [rsp+10h] [rbp-60h]
  __int64 v34; // [rsp+18h] [rbp-58h]
  __int64 *v35; // [rsp+20h] [rbp-50h]
  __int64 *v36; // [rsp+20h] [rbp-50h]
  __int64 *v37; // [rsp+28h] [rbp-48h]
  __int64 *v38; // [rsp+28h] [rbp-48h]
  __int64 v39; // [rsp+30h] [rbp-40h] BYREF
  int v40; // [rsp+38h] [rbp-38h]

  v6 = *(_QWORD **)(a2 + 32);
  v7 = *v6;
  v8 = v6[1];
  v9 = *(unsigned __int16 *)(*v6 + 24LL);
  v10 = v6[5];
  v11 = v6[6];
  v12 = v6[10];
  v13 = v6[11];
  v14 = *(_WORD *)(v10 + 24);
  if ( v9 != 32 && v9 != 10 || v14 == 32 || v14 == 10 )
  {
    v15 = v6[10];
    result = 0;
    if ( *(_WORD *)(v15 + 24) == 63 )
    {
      v20 = *(_QWORD *)(a2 + 72);
      v21 = *a1;
      v22 = *(const void ****)(a2 + 40);
      v23 = *(_DWORD *)(a2 + 60);
      v39 = v20;
      if ( v20 )
      {
        v31 = v22;
        v33 = v23;
        v35 = v21;
        sub_1623A60((__int64)&v39, v20, 2);
        v22 = v31;
        v23 = v33;
        v21 = v35;
      }
      *((_QWORD *)&v28 + 1) = v11;
      *(_QWORD *)&v28 = v10;
      *((_QWORD *)&v26 + 1) = v8;
      *(_QWORD *)&v26 = v7;
      v40 = *(_DWORD *)(a2 + 64);
      result = sub_1D37440(v21, 64, (__int64)&v39, v22, v23, (__int64)&v39, a3, a4, a5, v26, v28);
      if ( v39 )
      {
        v36 = result;
        sub_161E7C0((__int64)&v39, v39);
        return v36;
      }
    }
  }
  else
  {
    v17 = *(_QWORD *)(a2 + 72);
    v18 = *(const void ****)(a2 + 40);
    v19 = *(_DWORD *)(a2 + 60);
    v37 = *a1;
    v39 = v17;
    if ( v17 )
    {
      v29 = v18;
      v30 = v19;
      v32 = v12;
      v34 = v13;
      sub_1623A60((__int64)&v39, v17, 2);
      v18 = v29;
      v19 = v30;
      v12 = v32;
      v13 = v34;
    }
    *((_QWORD *)&v27 + 1) = v13;
    *(_QWORD *)&v27 = v12;
    *((_QWORD *)&v25 + 1) = v8;
    *(_QWORD *)&v25 = v7;
    *((_QWORD *)&v24 + 1) = v11;
    *(_QWORD *)&v24 = v10;
    v40 = *(_DWORD *)(a2 + 64);
    result = sub_1D37470(v37, 66, (__int64)&v39, v18, v19, (__int64)&v39, v24, v25, v27);
    if ( v39 )
    {
      v38 = result;
      sub_161E7C0((__int64)&v39, v39);
      return v38;
    }
  }
  return result;
}
