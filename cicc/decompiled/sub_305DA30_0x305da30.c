// Function: sub_305DA30
// Address: 0x305da30
//
__int64 __fastcall sub_305DA30(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 v4; // r8
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rbx
  _BYTE *v8; // rcx
  int v9; // eax
  _BYTE *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rax
  unsigned int v13; // edx
  unsigned int v14; // r12d
  __int64 v16; // [rsp+0h] [rbp-70h]
  __int64 v17; // [rsp+8h] [rbp-68h]
  _BYTE *v18; // [rsp+10h] [rbp-60h] BYREF
  __int64 v19; // [rsp+18h] [rbp-58h]
  _BYTE v20[80]; // [rsp+20h] [rbp-50h] BYREF

  v2 = 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v3 = *(_QWORD *)(a2 - 8);
    v4 = v3 + v2;
  }
  else
  {
    v4 = a2;
    v3 = a2 - v2;
  }
  v5 = v4 - v3;
  v18 = v20;
  v6 = v5 >> 5;
  v19 = 0x400000000LL;
  v7 = v5 >> 5;
  if ( (unsigned __int64)v5 > 0x80 )
  {
    v16 = v5;
    v17 = v5 >> 5;
    sub_C8D5F0((__int64)&v18, v20, v5 >> 5, 8u, v5, v6);
    v10 = v18;
    v9 = v19;
    LODWORD(v6) = v17;
    v5 = v16;
    v8 = &v18[8 * (unsigned int)v19];
  }
  else
  {
    v8 = v20;
    v9 = 0;
    v10 = v20;
  }
  if ( v5 > 0 )
  {
    v11 = 0;
    do
    {
      *(_QWORD *)&v8[v11] = *(_QWORD *)(v3 + 4 * v11);
      v11 += 8;
      --v7;
    }
    while ( v7 );
    v10 = v18;
    v9 = v19;
  }
  LODWORD(v19) = v6 + v9;
  v12 = sub_307A1F0(a1 + 8, a2, v10, (unsigned int)(v6 + v9), 3);
  if ( v13 )
    LODWORD(v3) = v13 >> 31;
  else
    LOBYTE(v3) = v12 <= 3;
  v14 = v3 ^ 1;
  if ( v18 != v20 )
    _libc_free((unsigned __int64)v18);
  return v14;
}
