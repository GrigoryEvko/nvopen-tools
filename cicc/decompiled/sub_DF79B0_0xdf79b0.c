// Function: sub_DF79B0
// Address: 0xdf79b0
//
unsigned __int64 __fastcall sub_DF79B0(__int64 *a1, __int64 *a2, __int64 a3, _BYTE *a4, _BYTE *a5, __int64 a6, int a7)
{
  __int64 *v7; // rbx
  unsigned __int64 v8; // r14
  __int64 *v9; // r12
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
  __int64 v25; // rcx
  int v26; // edx
  _QWORD *v27; // [rsp+0h] [rbp-A0h]
  _BYTE *v28; // [rsp+8h] [rbp-98h]
  __int64 v29; // [rsp+10h] [rbp-90h]
  int v30; // [rsp+10h] [rbp-90h]
  _BYTE *v33; // [rsp+30h] [rbp-70h] BYREF
  __int64 v34; // [rsp+38h] [rbp-68h]
  _BYTE v35[96]; // [rsp+40h] [rbp-60h] BYREF

  if ( a2 != &a2[a3] )
  {
    v7 = a2;
    v8 = 0;
    v9 = &a2[a3];
    while ( 1 )
    {
      while ( 1 )
      {
        v10 = (_BYTE *)*v7;
        if ( *(_BYTE *)*v7 == 63 )
          break;
LABEL_3:
        if ( v9 == ++v7 )
          return v8;
      }
      if ( (*a5 & 1) != 0 && v10 != a4 )
      {
        v29 = *v7;
        if ( !(unsigned __int8)sub_B4DD90(*v7) )
        {
          v11 = 1;
          if ( a7 == 1 )
          {
            v25 = *(_QWORD *)(v29 + 8);
            v26 = *(unsigned __int8 *)(v25 + 8);
            if ( (unsigned int)(v26 - 17) <= 1 )
              LOBYTE(v26) = *(_BYTE *)(**(_QWORD **)(v25 + 16) + 8LL);
            v11 = 3;
            if ( (unsigned __int8)v26 > 3u && (_BYTE)v26 != 5 )
              v11 = 2LL * ((v26 & 0xFD) == 4) + 1;
          }
          v12 = __OFADD__(v11, v8);
          v8 += v11;
          if ( v12 )
            v8 = 0x7FFFFFFFFFFFFFFFLL;
        }
        goto LABEL_3;
      }
      v13 = *((_DWORD *)v10 + 1);
      v33 = v35;
      v34 = 0x600000000LL;
      v14 = 32 * (1LL - (v13 & 0x7FFFFFF));
      v15 = &v10[v14];
      v16 = -v14;
      v17 = v16 >> 5;
      if ( (unsigned __int64)v16 > 0xC0 )
      {
        v27 = v15;
        v28 = v10;
        v30 = v16 >> 5;
        sub_C8D5F0((__int64)&v33, v35, v16 >> 5, 8u, v17, (__int64)v10);
        v20 = v33;
        v19 = v34;
        LODWORD(v17) = v30;
        v10 = v28;
        v15 = v27;
        v18 = &v33[8 * (unsigned int)v34];
      }
      else
      {
        v18 = v35;
        v19 = 0;
        v20 = v35;
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
        v20 = v33;
        v19 = v34;
      }
      v21 = v19 + v17;
      v22 = *((_QWORD *)v10 + 9);
      LODWORD(v34) = v21;
      v23 = sub_DF7390(a1, v22, *(_QWORD *)&v10[-32 * (*((_DWORD *)v10 + 1) & 0x7FFFFFF)], (__int64)v20, v21);
      v12 = __OFADD__(v23, v8);
      v8 += v23;
      if ( v12 )
      {
        v8 = 0x8000000000000000LL;
        if ( v23 > 0 )
          v8 = 0x7FFFFFFFFFFFFFFFLL;
      }
      if ( v33 == v35 )
        goto LABEL_3;
      _libc_free(v33, v22);
      if ( v9 == ++v7 )
        return v8;
    }
  }
  return 0;
}
