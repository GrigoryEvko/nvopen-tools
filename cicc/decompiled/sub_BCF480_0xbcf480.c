// Function: sub_BCF480
// Address: 0xbcf480
//
unsigned __int64 __fastcall sub_BCF480(__int64 *a1, const void *a2, __int64 a3, unsigned __int8 a4)
{
  __int64 v6; // rbx
  char v7; // al
  unsigned __int8 v8; // r8
  unsigned __int64 v9; // r14
  unsigned int v11; // esi
  int v12; // eax
  _QWORD *v13; // r9
  int v14; // eax
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rax
  unsigned __int8 v19; // [rsp+0h] [rbp-70h]
  unsigned __int64 *v20; // [rsp+8h] [rbp-68h]
  _QWORD *v21; // [rsp+8h] [rbp-68h]
  unsigned __int8 v22; // [rsp+8h] [rbp-68h]
  _QWORD *v23; // [rsp+10h] [rbp-60h] BYREF
  _QWORD *v24; // [rsp+18h] [rbp-58h] BYREF
  _QWORD v25[3]; // [rsp+20h] [rbp-50h] BYREF
  unsigned __int8 v26; // [rsp+38h] [rbp-38h]

  v6 = *(_QWORD *)*a1;
  v25[0] = a1;
  v25[1] = a2;
  v25[2] = a3;
  v26 = a4;
  v7 = sub_BCC6E0(v6 + 2904, (__int64)v25, &v23);
  v8 = a4;
  if ( v7 )
    return *v23;
  v11 = *(_DWORD *)(v6 + 2928);
  v12 = *(_DWORD *)(v6 + 2920);
  v13 = v23;
  ++*(_QWORD *)(v6 + 2904);
  v14 = v12 + 1;
  v24 = v13;
  if ( 4 * v14 >= 3 * v11 )
  {
    v11 *= 2;
    v22 = a4;
LABEL_14:
    sub_BCF240(v6 + 2904, v11);
    sub_BCC6E0(v6 + 2904, (__int64)v25, &v24);
    v13 = v24;
    v8 = v22;
    v14 = *(_DWORD *)(v6 + 2920) + 1;
    goto LABEL_6;
  }
  if ( v11 - *(_DWORD *)(v6 + 2924) - v14 <= v11 >> 3 )
  {
    v22 = a4;
    goto LABEL_14;
  }
LABEL_6:
  *(_DWORD *)(v6 + 2920) = v14;
  if ( *v13 != -4096 )
    --*(_DWORD *)(v6 + 2924);
  *v13 = 0;
  v15 = *(_QWORD *)(v6 + 2640);
  v16 = 8 * a3 + 32;
  *(_QWORD *)(v6 + 2720) += v16;
  v9 = (v15 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( *(_QWORD *)(v6 + 2648) >= v16 + v9 && v15 )
  {
    *(_QWORD *)(v6 + 2640) = v16 + v9;
  }
  else
  {
    v19 = v8;
    v21 = v13;
    v17 = sub_9D1E70(v6 + 2640, v16, 8 * a3 + 32, 3);
    v8 = v19;
    v13 = v21;
    v9 = v17;
  }
  v20 = v13;
  sub_BCB360(v9, a1, a2, a3, v8);
  *v20 = v9;
  return v9;
}
