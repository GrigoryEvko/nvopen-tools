// Function: sub_1D2EF30
// Address: 0x1d2ef30
//
__int64 __fastcall sub_1D2EF30(_QWORD *a1, __int64 a2, unsigned __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // r15
  char v8; // bl
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // r13
  _QWORD *v12; // r12
  _QWORD *v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  _BOOL8 v16; // rdi
  __int64 result; // rax
  unsigned __int64 v18; // r12
  __int64 v19; // r13
  __int64 v20; // rax
  unsigned __int8 *v21; // rsi
  __int64 v22; // r12
  unsigned __int64 v23; // rax
  __int64 v24; // rsi
  unsigned __int64 v25; // rax
  _QWORD *v26; // rdi
  _QWORD *v27; // [rsp+0h] [rbp-50h]
  __int64 v28; // [rsp+8h] [rbp-48h]
  unsigned __int8 *v29; // [rsp+18h] [rbp-38h] BYREF

  v6 = a3;
  v8 = a2;
  if ( (_BYTE)a2 )
  {
    v10 = a1[91];
    a3 = a1[90];
    v22 = (unsigned __int8)a2;
    v23 = (__int64)(v10 - a3) >> 3;
    if ( (unsigned __int8)a2 >= v23 )
    {
      v24 = (unsigned __int8)a2 + 1LL;
      if ( v22 + 1 > v23 )
      {
        sub_1D26750((__int64)(a1 + 90), v24 - v23);
        a3 = a1[90];
      }
      else if ( v22 + 1 < v23 )
      {
        v25 = a3 + 8 * v24;
        if ( v10 != v25 )
          a1[91] = v25;
      }
    }
    v18 = a3 + 8 * v22;
    result = *(_QWORD *)v18;
    if ( !*(_QWORD *)v18 )
    {
LABEL_9:
      v19 = a1[26];
      if ( v19 )
        a1[26] = *(_QWORD *)v19;
      else
        v19 = sub_145CBF0(a1 + 27, 112, 8);
      v20 = sub_1D274F0(1u, a3, v10, a5, a6);
      v29 = 0;
      *(_QWORD *)v19 = 0;
      v21 = v29;
      *(_QWORD *)(v19 + 40) = v20;
      *(_QWORD *)(v19 + 8) = 0;
      *(_QWORD *)(v19 + 16) = 0;
      *(_WORD *)(v19 + 24) = 6;
      *(_DWORD *)(v19 + 28) = -1;
      *(_QWORD *)(v19 + 32) = 0;
      *(_QWORD *)(v19 + 48) = 0;
      *(_QWORD *)(v19 + 56) = 0x100000000LL;
      *(_DWORD *)(v19 + 64) = 0;
      *(_QWORD *)(v19 + 72) = v21;
      if ( v21 )
        sub_1623210((__int64)&v29, v21, v19 + 72);
      *(_WORD *)(v19 + 80) &= 0xF000u;
      *(_WORD *)(v19 + 26) = 0;
      *(_BYTE *)(v19 + 88) = v8;
      *(_QWORD *)(v19 + 96) = v6;
      *(_QWORD *)v18 = v19;
      sub_1D172A0((__int64)a1, v19);
      return *(_QWORD *)v18;
    }
  }
  else
  {
    v9 = a1[95];
    v10 = (__int64)(a1 + 94);
    v11 = a2;
    v12 = a1 + 94;
    if ( !v9 )
      goto LABEL_3;
    do
    {
      while ( *(_BYTE *)(v9 + 32) || *(_QWORD *)(v9 + 40) >= a3 )
      {
        v12 = (_QWORD *)v9;
        v9 = *(_QWORD *)(v9 + 16);
        if ( !v9 )
          goto LABEL_18;
      }
      v9 = *(_QWORD *)(v9 + 24);
    }
    while ( v9 );
LABEL_18:
    if ( (_QWORD *)v10 == v12 || *((_BYTE *)v12 + 32) || v12[5] > a3 )
    {
LABEL_3:
      v27 = a1 + 94;
      LOBYTE(v11) = 0;
      v28 = (__int64)v12;
      v13 = (_QWORD *)sub_22077B0(56);
      v13[4] = v11;
      v12 = v13;
      v13[5] = v6;
      v13[6] = 0;
      v14 = sub_1D2EDB0(a1 + 93, v28, (__int64)(v13 + 4));
      if ( v15 )
      {
        if ( v27 == (_QWORD *)v15 || v14 )
        {
          v16 = 1;
        }
        else
        {
          v16 = 1;
          if ( !*(_BYTE *)(v15 + 32) )
            v16 = v6 < *(_QWORD *)(v15 + 40);
        }
        sub_220F040(v16, v12, v15, v27);
        ++a1[98];
      }
      else
      {
        v26 = v12;
        v12 = (_QWORD *)v14;
        j_j___libc_free_0(v26, 56);
      }
    }
    result = v12[6];
    v18 = (unsigned __int64)(v12 + 6);
    if ( !result )
      goto LABEL_9;
  }
  return result;
}
