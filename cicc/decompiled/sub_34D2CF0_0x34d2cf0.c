// Function: sub_34D2CF0
// Address: 0x34d2cf0
//
unsigned __int64 __fastcall sub_34D2CF0(
        __int64 *a1,
        __int64 *a2,
        __int64 a3,
        _BYTE *a4,
        _BYTE *a5,
        unsigned __int64 a6,
        int a7)
{
  __int64 *v7; // r12
  __int64 *v8; // rbx
  unsigned __int64 v9; // r15
  _BYTE *v10; // r9
  __int64 v11; // rax
  bool v12; // of
  int v13; // eax
  __int64 v14; // rdx
  _QWORD *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r8
  _QWORD *v18; // rdx
  int v19; // esi
  _BYTE *v20; // rcx
  unsigned int v21; // r8d
  __int64 v22; // rsi
  __int64 v23; // rax
  _QWORD *v25; // [rsp+8h] [rbp-A8h]
  _BYTE *v26; // [rsp+10h] [rbp-A0h]
  int v27; // [rsp+18h] [rbp-98h]
  __int64 v28; // [rsp+20h] [rbp-90h]
  _BYTE *v32; // [rsp+40h] [rbp-70h] BYREF
  __int64 v33; // [rsp+48h] [rbp-68h]
  _BYTE v34[96]; // [rsp+50h] [rbp-60h] BYREF

  v7 = &a2[a3];
  if ( v7 != a2 )
  {
    v8 = a2;
    v9 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v10 = (_BYTE *)*v8;
        if ( *(_BYTE *)*v8 == 63 )
          break;
LABEL_3:
        if ( v7 == ++v8 )
          return v9;
      }
      if ( (*a5 & 1) != 0 && v10 != a4 )
      {
        v28 = *v8;
        if ( !(unsigned __int8)sub_B4DD90(*v8) )
        {
          v11 = sub_34D2250((__int64)(a1 - 1), 0xDu, *(_QWORD *)(v28 + 8), a7, 0, 0, 0, 0, 0);
          v12 = __OFADD__(v11, v9);
          v9 += v11;
          if ( v12 )
          {
            v9 = 0x8000000000000000LL;
            if ( v11 > 0 )
              v9 = 0x7FFFFFFFFFFFFFFFLL;
          }
        }
        goto LABEL_3;
      }
      v13 = *((_DWORD *)v10 + 1);
      v32 = v34;
      v33 = 0x600000000LL;
      v14 = 32 * (1LL - (v13 & 0x7FFFFFF));
      v15 = &v10[v14];
      v16 = -v14;
      v17 = v16 >> 5;
      if ( (unsigned __int64)v16 > 0xC0 )
      {
        v25 = v15;
        v26 = v10;
        v27 = v16 >> 5;
        sub_C8D5F0((__int64)&v32, v34, v16 >> 5, 8u, v17, (__int64)v10);
        v20 = v32;
        v19 = v33;
        LODWORD(v17) = v27;
        v10 = v26;
        v15 = v25;
        v18 = &v32[8 * (unsigned int)v33];
      }
      else
      {
        v18 = v34;
        v19 = 0;
        v20 = v34;
      }
      if ( v10 != (_BYTE *)v15 )
      {
        do
        {
          if ( v18 )
            *v18 = *v15;
          v15 += 4;
          ++v18;
        }
        while ( v10 != (_BYTE *)v15 );
        v20 = v32;
        v19 = v33;
      }
      v21 = v19 + v17;
      v22 = *((_QWORD *)v10 + 9);
      LODWORD(v33) = v21;
      v23 = sub_34D1940(a1, v22, *(_QWORD *)&v10[-32 * (*((_DWORD *)v10 + 1) & 0x7FFFFFF)], (__int64)v20, v21, a6);
      v12 = __OFADD__(v23, v9);
      v9 += v23;
      if ( v12 )
      {
        v9 = 0x8000000000000000LL;
        if ( v23 > 0 )
          v9 = 0x7FFFFFFFFFFFFFFFLL;
      }
      if ( v32 == v34 )
        goto LABEL_3;
      _libc_free((unsigned __int64)v32);
      if ( v7 == ++v8 )
        return v9;
    }
  }
  return 0;
}
