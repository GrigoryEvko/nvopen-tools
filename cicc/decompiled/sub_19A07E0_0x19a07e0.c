// Function: sub_19A07E0
// Address: 0x19a07e0
//
__int64 __fastcall sub_19A07E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  int v7; // eax
  _QWORD *v8; // rax
  unsigned __int64 v9; // rdi
  __int64 v10; // r13
  unsigned __int64 *v11; // r15
  unsigned __int64 v12; // rax
  unsigned __int64 *v13; // r13
  unsigned __int64 v14; // rcx
  unsigned __int64 v15; // rdx
  unsigned __int64 *v16; // rax
  unsigned __int64 *v17; // rsi
  unsigned int v18; // r12d
  int v20; // r8d
  int v21; // r9d
  unsigned __int64 *v22; // rsi
  __int64 v23; // [rsp+18h] [rbp-68h] BYREF
  _BYTE *v24; // [rsp+20h] [rbp-60h] BYREF
  __int64 v25; // [rsp+28h] [rbp-58h]
  _BYTE v26[80]; // [rsp+30h] [rbp-50h] BYREF

  v25 = 0x400000000LL;
  v7 = *(_DWORD *)(a2 + 40);
  v24 = v26;
  if ( !v7 )
  {
    v8 = v26;
    if ( !*(_QWORD *)(a2 + 80) )
      goto LABEL_10;
    goto LABEL_3;
  }
  sub_19930D0((__int64)&v24, a2 + 32, a3, a4, a5, a6);
  if ( *(_QWORD *)(a2 + 80) )
  {
    if ( (unsigned int)v25 >= HIDWORD(v25) )
      sub_16CD150((__int64)&v24, v26, 0, 8, v20, v21);
    v8 = &v24[8 * (unsigned int)v25];
LABEL_3:
    *v8 = *(_QWORD *)(a2 + 80);
    v9 = (unsigned __int64)v24;
    LODWORD(v25) = v25 + 1;
    v10 = 8LL * (unsigned int)v25;
    v11 = (unsigned __int64 *)&v24[v10];
    goto LABEL_4;
  }
  v9 = (unsigned __int64)v24;
  v10 = 8LL * (unsigned int)v25;
  v11 = (unsigned __int64 *)&v24[v10];
LABEL_4:
  if ( (unsigned __int64 *)v9 != v11 )
  {
    _BitScanReverse64(&v12, v10 >> 3);
    sub_1993A10((char *)v9, v11, 2LL * (int)(63 - (v12 ^ 0x3F)));
    if ( (unsigned __int64)v10 <= 0x80 )
    {
      sub_1992E50((unsigned __int64 *)v9, v11);
    }
    else
    {
      v13 = (unsigned __int64 *)(v9 + 128);
      sub_1992E50((unsigned __int64 *)v9, (unsigned __int64 *)(v9 + 128));
      if ( (unsigned __int64 *)(v9 + 128) != v11 )
      {
        do
        {
          while ( 1 )
          {
            v14 = *v13;
            v15 = *(v13 - 1);
            v16 = v13 - 1;
            if ( v15 > *v13 )
              break;
            v22 = v13++;
            *v22 = v14;
            if ( v13 == v11 )
              goto LABEL_10;
          }
          do
          {
            v16[1] = v15;
            v17 = v16;
            v15 = *--v16;
          }
          while ( v14 < v15 );
          ++v13;
          *v17 = v14;
        }
        while ( v13 != v11 );
      }
    }
  }
LABEL_10:
  v18 = sub_19A0320(a1, (__int64)&v24, &v23);
  if ( v24 != v26 )
    _libc_free((unsigned __int64)v24);
  return v18;
}
