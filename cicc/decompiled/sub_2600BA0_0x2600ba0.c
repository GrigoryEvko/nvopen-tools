// Function: sub_2600BA0
// Address: 0x2600ba0
//
__int64 __fastcall sub_2600BA0(__int64 *a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdi
  int v4; // eax
  _QWORD *v5; // rdi
  __int64 v6; // r12
  __int64 v7; // rsi
  __int64 v8; // rdi
  __int64 result; // rax
  _QWORD *v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rdi
  __int64 v15; // [rsp+0h] [rbp-30h] BYREF
  __int64 v16; // [rsp+8h] [rbp-28h]
  __int64 v17; // [rsp+10h] [rbp-20h]
  __int64 v18; // [rsp+18h] [rbp-18h]

  if ( **(_BYTE **)(*(_QWORD *)(*a1 + 8) + 16LL) == 84 && !(unsigned __int8)sub_AA5590(a1[33], 0) )
  {
    v11 = sub_AA54C0(a1[33]);
    sub_AA5DE0(a1[33], a1[33], v11);
  }
  v2 = a1[33];
  v3 = *(_QWORD *)(v2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v3 == v2 + 48 )
  {
    v5 = 0;
  }
  else
  {
    if ( !v3 )
      BUG();
    v4 = *(unsigned __int8 *)(v3 - 24);
    v5 = (_QWORD *)(v3 - 24);
    if ( (unsigned int)(v4 - 30) >= 0xB )
      v5 = 0;
  }
  sub_B43D60(v5);
  if ( !a1[31] )
  {
    v12 = *a1;
    v17 = 0;
    v18 = 0;
    v13 = *(_QWORD *)(v12 + 16);
    v16 = 0;
    v14 = *(_QWORD *)(v12 + 8);
    v15 = 0;
    sub_25FFEC0(v14, v13, (__int64)&v15);
    sub_25F7360(a1[34], a1[34], a1[33], (__int64)&v15);
    if ( !*((_BYTE *)a1 + 129) )
      sub_25F7360(a1[36], a1[36], a1[35], (__int64)&v15);
    sub_C7D6A0(v16, 8LL * (unsigned int)v18, 8);
  }
  sub_AA80F0(
    a1[33],
    (unsigned __int64 *)(a1[33] + 48),
    0,
    a1[34],
    *(__int64 **)(a1[34] + 56),
    1,
    (__int64 *)(a1[34] + 48),
    0);
  v6 = a1[35];
  v7 = a1[34];
  v8 = a1[33];
  if ( v7 == v6 )
    v6 = a1[33];
  if ( !*((_BYTE *)a1 + 129) )
  {
    if ( sub_AA5780(v6) )
    {
      v10 = (_QWORD *)sub_986580(v6);
      sub_B43D60(v10);
      sub_AA80F0(
        v6,
        (unsigned __int64 *)(v6 + 48),
        0,
        a1[36],
        *(__int64 **)(a1[36] + 56),
        1,
        (__int64 *)(a1[36] + 48),
        0);
      sub_AA5DE0(v6, a1[36], v6);
      sub_AA5450((_QWORD *)a1[36]);
    }
    v8 = a1[33];
    v7 = a1[34];
  }
  sub_AA5DE0(v8, v7, v8);
  sub_AA5450((_QWORD *)a1[34]);
  result = a1[33];
  a1[35] = 0;
  a1[33] = 0;
  a1[34] = result;
  a1[36] = 0;
  *((_BYTE *)a1 + 256) = 0;
  return result;
}
