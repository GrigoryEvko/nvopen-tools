// Function: sub_33ED040
// Address: 0x33ed040
//
__int64 __fastcall sub_33ED040(_QWORD *a1, unsigned int a2)
{
  __int64 v2; // r13
  __int64 v4; // rax
  unsigned __int64 v5; // rdx
  __int64 *v6; // rax
  __int64 v7; // r8
  unsigned __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // r14
  __int64 v12; // rax
  unsigned __int8 *v13; // rsi
  __int64 v14; // rsi
  unsigned __int64 v15; // rdx
  unsigned __int8 *v16; // [rsp+8h] [rbp-38h] BYREF

  v2 = a2;
  v4 = a1[103];
  v5 = (a1[104] - v4) >> 3;
  if ( a2 >= v5 )
  {
    v9 = (int)(a2 + 1);
    if ( v9 > v5 )
    {
      sub_33E4560((__int64)(a1 + 103), v9 - v5);
      v4 = a1[103];
    }
    else if ( v9 < v5 )
    {
      v10 = v4 + 8 * v9;
      if ( a1[104] != v10 )
        a1[104] = v10;
    }
  }
  v6 = (__int64 *)(8 * v2 + v4);
  v7 = *v6;
  if ( !*v6 )
  {
    v11 = a1[52];
    if ( v11 )
    {
      a1[52] = *(_QWORD *)v11;
    }
    else
    {
      v14 = a1[53];
      a1[63] += 120LL;
      v15 = (v14 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      if ( a1[54] >= v15 + 120 && v14 )
      {
        a1[53] = v15 + 120;
        if ( !v15 )
        {
LABEL_13:
          *v6 = v11;
          sub_33CC420((__int64)a1, v11);
          return *(_QWORD *)(a1[103] + 8 * v2);
        }
        v11 = (v14 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      }
      else
      {
        v11 = sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
      }
    }
    v12 = sub_33ECD10(1u);
    v16 = 0;
    *(_QWORD *)v11 = 0;
    v13 = v16;
    *(_QWORD *)(v11 + 48) = v12;
    *(_QWORD *)(v11 + 8) = 0;
    *(_QWORD *)(v11 + 16) = 0;
    *(_QWORD *)(v11 + 24) = 8;
    *(_WORD *)(v11 + 34) = -1;
    *(_DWORD *)(v11 + 36) = -1;
    *(_QWORD *)(v11 + 40) = 0;
    *(_QWORD *)(v11 + 56) = 0;
    *(_QWORD *)(v11 + 64) = 0x100000000LL;
    *(_DWORD *)(v11 + 72) = 0;
    *(_QWORD *)(v11 + 80) = v13;
    if ( v13 )
      sub_B976B0((__int64)&v16, v13, v11 + 80);
    *(_QWORD *)(v11 + 88) = 0xFFFFFFFFLL;
    *(_WORD *)(v11 + 32) = 0;
    *(_DWORD *)(v11 + 96) = a2;
    v6 = (__int64 *)(8 * v2 + a1[103]);
    goto LABEL_13;
  }
  return v7;
}
