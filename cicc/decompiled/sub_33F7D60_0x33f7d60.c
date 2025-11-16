// Function: sub_33F7D60
// Address: 0x33f7d60
//
__int64 __fastcall sub_33F7D60(_QWORD *a1, __int64 a2, unsigned __int64 a3)
{
  __int16 v5; // bx
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // r12
  _QWORD *v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  char v12; // di
  __int64 result; // rax
  __int64 v14; // r12
  __int64 v15; // r13
  __int64 v16; // rax
  unsigned __int8 *v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // r12
  unsigned __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rax
  unsigned __int64 v23; // rdi
  __int64 v24; // rcx
  unsigned __int64 v25; // rax
  _QWORD *v26; // [rsp+0h] [rbp-50h]
  __int64 v27; // [rsp+8h] [rbp-48h]
  unsigned __int8 *v28; // [rsp+18h] [rbp-38h] BYREF

  v5 = a2;
  if ( (_WORD)a2 )
  {
    v18 = a1[106];
    v19 = (unsigned __int16)a2;
    v20 = (a1[107] - v18) >> 3;
    if ( (unsigned __int16)a2 >= v20 )
    {
      v21 = (unsigned __int16)a2 + 1LL;
      if ( v19 + 1 > v20 )
      {
        sub_33E43B0((__int64)(a1 + 106), v21 - v20);
        v18 = a1[106];
      }
      else if ( v19 + 1 < v20 )
      {
        v22 = v18 + 8 * v21;
        if ( a1[107] != v22 )
          a1[107] = v22;
      }
    }
    v14 = v18 + 8 * v19;
    result = *(_QWORD *)v14;
    if ( !*(_QWORD *)v14 )
      goto LABEL_9;
  }
  else
  {
    v6 = a1[111];
    v7 = a2;
    v8 = (__int64)(a1 + 110);
    if ( !v6 )
      goto LABEL_3;
    do
    {
      while ( *(_WORD *)(v6 + 32) || *(_QWORD *)(v6 + 40) >= a3 )
      {
        v8 = v6;
        v6 = *(_QWORD *)(v6 + 16);
        if ( !v6 )
          goto LABEL_19;
      }
      v6 = *(_QWORD *)(v6 + 24);
    }
    while ( v6 );
LABEL_19:
    if ( a1 + 110 == (_QWORD *)v8 || *(_WORD *)(v8 + 32) || *(_QWORD *)(v8 + 40) > a3 )
    {
LABEL_3:
      v26 = a1 + 110;
      LOWORD(v7) = 0;
      v27 = v8;
      v9 = (_QWORD *)sub_22077B0(0x38u);
      v9[4] = v7;
      v8 = (__int64)v9;
      v9[5] = a3;
      v9[6] = 0;
      v10 = sub_33F7BE0(a1 + 109, v27, (__int64)(v9 + 4));
      if ( v11 )
      {
        if ( v26 == (_QWORD *)v11 || v10 )
        {
          v12 = 1;
        }
        else
        {
          v12 = 1;
          if ( !*(_WORD *)(v11 + 32) )
            v12 = a3 < *(_QWORD *)(v11 + 40);
        }
        sub_220F040(v12, v8, (_QWORD *)v11, v26);
        ++a1[114];
      }
      else
      {
        v23 = v8;
        v8 = v10;
        j_j___libc_free_0(v23);
      }
    }
    result = *(_QWORD *)(v8 + 48);
    v14 = v8 + 48;
    if ( !result )
    {
LABEL_9:
      v15 = a1[52];
      if ( v15 )
      {
        a1[52] = *(_QWORD *)v15;
      }
      else
      {
        v24 = a1[53];
        a1[63] += 120LL;
        v25 = (v24 + 7) & 0xFFFFFFFFFFFFFFF8LL;
        if ( a1[54] >= v25 + 120 && v24 )
        {
          a1[53] = v25 + 120;
          if ( !v25 )
          {
LABEL_14:
            *(_QWORD *)v14 = v15;
            sub_33CC420((__int64)a1, v15);
            return *(_QWORD *)v14;
          }
        }
        else
        {
          v25 = sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
        }
        v15 = v25;
      }
      v16 = sub_33ECD10(1u);
      v28 = 0;
      *(_QWORD *)v15 = 0;
      v17 = v28;
      *(_QWORD *)(v15 + 48) = v16;
      *(_QWORD *)(v15 + 8) = 0;
      *(_QWORD *)(v15 + 16) = 0;
      *(_QWORD *)(v15 + 24) = 7;
      *(_WORD *)(v15 + 34) = -1;
      *(_DWORD *)(v15 + 36) = -1;
      *(_QWORD *)(v15 + 40) = 0;
      *(_QWORD *)(v15 + 56) = 0;
      *(_QWORD *)(v15 + 64) = 0x100000000LL;
      *(_DWORD *)(v15 + 72) = 0;
      *(_QWORD *)(v15 + 80) = v17;
      if ( v17 )
        sub_B976B0((__int64)&v28, v17, v15 + 80);
      *(_QWORD *)(v15 + 88) = 0xFFFFFFFFLL;
      *(_WORD *)(v15 + 32) = 0;
      *(_WORD *)(v15 + 96) = v5;
      *(_QWORD *)(v15 + 104) = a3;
      goto LABEL_14;
    }
  }
  return result;
}
