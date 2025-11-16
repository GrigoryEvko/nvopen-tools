// Function: sub_636E20
// Address: 0x636e20
//
__int64 __fastcall sub_636E20(__int64 *a1, __int64 *a2, __m128i *a3, __int64 a4, __int64 a5)
{
  __int64 *v5; // r10
  __int64 v8; // r12
  __int64 v9; // r15
  __int64 v10; // r11
  __int64 result; // rax
  __int64 v12; // rdi
  int v13; // eax
  bool v14; // zf
  __int64 v15; // [rsp+0h] [rbp-60h]
  __int64 v16; // [rsp+8h] [rbp-58h]
  __int64 v17; // [rsp+10h] [rbp-50h]
  __int64 v18[7]; // [rsp+28h] [rbp-38h] BYREF

  v5 = a1;
  v8 = *a2;
  v9 = *(_QWORD *)(*(_QWORD *)(*a2 + 40) + 32LL);
  v10 = *(_QWORD *)(*a2 + 120);
  if ( (a3[2].m128i_i8[9] & 0x20) != 0 )
    v10 = *(_QWORD *)&dword_4D03B80;
  if ( *(_QWORD *)(v8 + 112) && *(_BYTE *)(v9 + 140) != 11
    || (v16 = a5, v17 = v10, v15 = *a1, v13 = sub_8D2430(*(_QWORD *)(*a2 + 120)), v5 = a1, v10 = v17, a5 = v16, !v13) )
  {
    result = sub_634B10(v5, v10, v8, a3, a5, v18);
    if ( (a3[2].m128i_i8[8] & 0x40) != 0 )
      goto LABEL_8;
    v12 = v18[0];
    if ( !v18[0] )
      goto LABEL_8;
LABEL_7:
    result = sub_72A690(v12, a4, 0, v8);
    goto LABEL_8;
  }
  result = sub_62FA90(v15, v8, (__int64)a3);
  if ( !(_DWORD)result )
  {
    *a1 = 0;
    v14 = *(_BYTE *)(v9 + 140) == 11;
    v18[0] = 0;
    if ( v14 )
      goto LABEL_14;
    goto LABEL_9;
  }
  result = sub_634B10(a1, v17, v8, a3, v16, v18);
  v12 = v18[0];
  if ( v18[0] )
  {
    *(_BYTE *)(v18[0] + 170) |= 1u;
    if ( (a3[2].m128i_i8[8] & 0x40) == 0 )
      goto LABEL_7;
  }
LABEL_8:
  if ( *(_BYTE *)(v9 + 140) == 11 )
  {
LABEL_14:
    *a2 = 0;
    return result;
  }
LABEL_9:
  if ( (a3[2].m128i_i8[9] & 0x20) == 0 )
  {
    result = sub_72FD90(*(_QWORD *)(v8 + 112), 7);
    *a2 = result;
  }
  return result;
}
