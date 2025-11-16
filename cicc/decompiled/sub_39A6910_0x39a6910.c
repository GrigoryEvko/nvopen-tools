// Function: sub_39A6910
// Address: 0x39a6910
//
__int64 __fastcall sub_39A6910(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r14
  unsigned int v5; // eax
  __int64 v6; // rdx
  __int16 v7; // ax
  __int64 v8; // r8
  __int64 result; // rax
  __int64 v10; // rdx
  int v11[9]; // [rsp+Ch] [rbp-24h] BYREF

  v4 = *(_QWORD *)(a3 + 8 * (3LL - *(unsigned int *)(a3 + 8)));
  if ( !v4 )
  {
    v6 = 0;
LABEL_4:
    sub_39A6830(a1, a2, v6);
    v7 = *(_WORD *)(a1[10] + 24);
    if ( ((v7 - 12) & 0xFFFB) == 0 )
      goto LABEL_15;
    goto LABEL_5;
  }
  v5 = *(_DWORD *)(v4 + 8);
  if ( !v5 )
    goto LABEL_3;
  v10 = *(_QWORD *)(v4 - 8LL * v5);
  if ( v10 )
  {
    sub_39A6760(a1, a2, v10, 73);
    v5 = *(_DWORD *)(v4 + 8);
  }
  if ( v5 != 2 )
  {
LABEL_3:
    v6 = v4;
    goto LABEL_4;
  }
  if ( !*(_QWORD *)(v4 - 8) )
  {
    sub_39A6830(a1, a2, v4);
    goto LABEL_6;
  }
  sub_39A6830(a1, a2, v4);
  v7 = *(_WORD *)(a1[10] + 24);
  if ( ((v7 - 12) & 0xFFFB) != 0 )
  {
LABEL_5:
    if ( v7 == 1 )
      goto LABEL_15;
LABEL_6:
    v8 = *(unsigned __int8 *)(a3 + 52);
    if ( (unsigned __int8)v8 > 1u )
      goto LABEL_16;
LABEL_7:
    result = *(unsigned int *)(a3 + 28);
    if ( (result & 0x2000) != 0 )
      goto LABEL_17;
    goto LABEL_8;
  }
LABEL_15:
  sub_39A34D0((__int64)a1, a2, 39);
  v8 = *(unsigned __int8 *)(a3 + 52);
  if ( (unsigned __int8)v8 <= 1u )
    goto LABEL_7;
LABEL_16:
  v11[0] = 65547;
  sub_39A3560((__int64)a1, (__int64 *)(a2 + 8), 54, (__int64)v11, v8);
  result = *(unsigned int *)(a3 + 28);
  if ( (result & 0x2000) != 0 )
  {
LABEL_17:
    sub_39A34D0((__int64)a1, a2, 119);
    result = *(unsigned int *)(a3 + 28);
    if ( (result & 0x4000) != 0 )
      return sub_39A34D0((__int64)a1, a2, 120);
    return result;
  }
LABEL_8:
  if ( (result & 0x4000) != 0 )
    return sub_39A34D0((__int64)a1, a2, 120);
  return result;
}
