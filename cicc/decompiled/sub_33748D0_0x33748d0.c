// Function: sub_33748D0
// Address: 0x33748d0
//
__int64 __fastcall sub_33748D0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r13
  __int64 v9; // rax
  int v10; // edx
  __int64 v11; // r11
  __int64 v12; // r12
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // r15
  __int64 v16; // rax
  int v17; // eax
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v21; // [rsp+8h] [rbp-78h]
  __int64 v25; // [rsp+40h] [rbp-40h] BYREF
  int v26; // [rsp+48h] [rbp-38h]

  v8 = *(_QWORD *)(a1[108] + 40);
  v9 = sub_E6C430(*(_QWORD *)(v8 + 24), a2, a3, a4, a5);
  v10 = *((_DWORD *)a1 + 212);
  v11 = a1[108];
  v25 = 0;
  v12 = v9;
  v13 = *a1;
  v26 = v10;
  if ( v13 )
  {
    if ( &v25 != (__int64 *)(v13 + 48) )
    {
      v14 = *(_QWORD *)(v13 + 48);
      v25 = v14;
      if ( v14 )
      {
        v21 = v11;
        sub_B96E90((__int64)&v25, v14, 1);
        v11 = v21;
      }
    }
  }
  v15 = sub_33F2D10(v11, &v25, a2, a3, v12);
  if ( v25 )
    sub_B91220((__int64)&v25, v25);
  v16 = sub_B2E500(*(_QWORD *)a1[120]);
  v17 = sub_B2A630(v16);
  if ( *(_BYTE *)(v8 + 580) && (unsigned int)(v17 - 7) <= 3 )
  {
    sub_3017890(*(_QWORD *)(v8 + 88), a4, a6, v12);
    return v15;
  }
  if ( v17 > 10 )
  {
    if ( v17 == 12 )
      return v15;
    goto LABEL_11;
  }
  if ( v17 <= 6 )
LABEL_11:
    sub_2E7D2B0(
      (unsigned __int64 *)v8,
      *(_QWORD *)(*(_QWORD *)(a1[120] + 56) + 8LL * *(unsigned int *)(a5 + 44)),
      a6,
      v12,
      v18,
      v19);
  return v15;
}
