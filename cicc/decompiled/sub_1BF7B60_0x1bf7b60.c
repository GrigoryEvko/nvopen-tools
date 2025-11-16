// Function: sub_1BF7B60
// Address: 0x1bf7b60
//
__int64 __fastcall sub_1BF7B60(__int64 *a1, unsigned __int8 a2, __m128i a3, __m128i a4)
{
  __int64 *v5; // r13
  __int64 v6; // rax
  __int64 v7; // rsi
  char v8; // r14
  unsigned int v9; // r13d
  __int64 v11; // rdx
  unsigned int v12; // ebx
  __int64 v13; // rbx
  _QWORD *v14; // r13
  char *v15; // rax
  char *v16; // rbx
  char *v17; // r12
  char *v18; // rdi
  __int64 v19; // rax
  __int64 v20; // r13
  _QWORD v21[11]; // [rsp+0h] [rbp-210h] BYREF
  char *v22; // [rsp+58h] [rbp-1B8h]
  unsigned int v23; // [rsp+60h] [rbp-1B0h]
  char v24; // [rsp+68h] [rbp-1A8h] BYREF

  v5 = (__int64 *)a1[7];
  v6 = sub_15E0530(*v5);
  if ( sub_1602790(v6)
    || (v19 = sub_15E0530(*v5),
        v20 = sub_16033E0(v19),
        (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v20 + 32LL))(
          v20,
          "loop-vectorize",
          14))
    || (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v20 + 40LL))(
         v20,
         "loop-vectorize",
         14)
    || (v8 = (*(__int64 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v20 + 24LL))(
               v20,
               "loop-vectorize",
               14)) != 0 )
  {
    v7 = *a1;
    v8 = 1;
    v9 = sub_1BF3B30(a1, *a1, a2);
  }
  else
  {
    v7 = *a1;
    v9 = sub_1BF3B30(a1, *a1, a2);
    if ( !(_BYTE)v9 )
      return 0;
  }
  if ( *(_QWORD *)(*a1 + 16) == *(_QWORD *)(*a1 + 8) )
  {
    if ( (unsigned int)((__int64)(*(_QWORD *)(*a1 + 40) - *(_QWORD *)(*a1 + 32)) >> 3) == 1
      || (unsigned __int8)sub_1BF2D30(a1) )
    {
      if ( !(unsigned __int8)sub_1BF6690(a1, a3, a4) )
      {
        if ( !v8 )
          return 0;
        v9 = 0;
      }
    }
    else
    {
      if ( !v8 )
        return 0;
      v9 = 0;
      sub_1BF6690(a1, a3, a4);
    }
    if ( !(unsigned __int8)sub_1BF3C70((__int64)a1, a3, a4, v7, v11) )
    {
      if ( !v8 )
        return 0;
      v9 = 0;
    }
    v12 = dword_4FB98C0;
    if ( *(_DWORD *)(a1[58] + 40) == 1 )
      v12 = dword_4FB97E0;
    if ( v12 < *(_DWORD *)(sub_1458800(a1[2]) + 48) )
    {
      v13 = *a1;
      v14 = (_QWORD *)a1[7];
      v15 = sub_1BF18B0(a1[58]);
      sub_1BF1750((__int64)v21, (__int64)v15, (__int64)"TooManySCEVRunTimeChecks", 24, v13, 0);
      sub_15CAB20((__int64)v21, "Too many SCEV assumptions need to be made and checked ", 0x36u);
      sub_15CAB20((__int64)v21, "at runtime", 0xAu);
      sub_143AA50(v14, (__int64)v21);
      v16 = v22;
      v21[0] = &unk_49ECF68;
      v17 = &v22[88 * v23];
      if ( v22 != v17 )
      {
        do
        {
          v17 -= 88;
          v18 = (char *)*((_QWORD *)v17 + 4);
          if ( v18 != v17 + 48 )
            j_j___libc_free_0(v18, *((_QWORD *)v17 + 6) + 1LL);
          if ( *(char **)v17 != v17 + 16 )
            j_j___libc_free_0(*(_QWORD *)v17, *((_QWORD *)v17 + 2) + 1LL);
        }
        while ( v16 != v17 );
        v17 = v22;
      }
      if ( v17 != &v24 )
        _libc_free((unsigned __int64)v17);
      return 0;
    }
  }
  else if ( !(unsigned __int8)sub_1BF21A0(a1) )
  {
    return 0;
  }
  return v9;
}
