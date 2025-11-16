// Function: sub_BB6360
// Address: 0xbb6360
//
__int64 __fastcall sub_BB6360(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int8 (__fastcall *a4)(__int64, __int64, __int64 *),
        __int64 a5)
{
  __int64 v6; // r10
  __int64 v10; // rdx
  __int64 v11; // r8
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // r11
  __int64 v15; // rbx
  _BYTE *v16; // rcx
  int v17; // edx
  _QWORD *v18; // rsi
  __int64 v19; // rax
  unsigned int v20; // edx
  __int64 v21; // rbx
  __int64 v22; // rax
  unsigned int v23; // r12d
  __int64 v25; // [rsp+8h] [rbp-98h]
  __int64 v26; // [rsp+18h] [rbp-88h]
  __int64 v27; // [rsp+20h] [rbp-80h]
  __int64 v28; // [rsp+20h] [rbp-80h]
  __int64 v29; // [rsp+28h] [rbp-78h]
  _BYTE *v30; // [rsp+30h] [rbp-70h] BYREF
  __int64 v31; // [rsp+38h] [rbp-68h]
  _BYTE v32[96]; // [rsp+40h] [rbp-60h] BYREF

  v6 = a1;
  v10 = 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
  {
    v11 = *(_QWORD *)(a1 - 8);
    v12 = v11 + v10;
  }
  else
  {
    v12 = a1;
    v11 = a1 - v10;
  }
  v13 = v12 - (v11 + 32);
  v30 = v32;
  v31 = 0x600000000LL;
  v14 = v13 >> 5;
  v15 = v13 >> 5;
  if ( (unsigned __int64)v13 > 0xC0 )
  {
    v28 = v11;
    v25 = a3;
    v26 = v13;
    v29 = v13 >> 5;
    sub_C8D5F0(&v30, v32, v13 >> 5, 8);
    v18 = v30;
    v17 = v31;
    LODWORD(v14) = v29;
    v11 = v28;
    v13 = v26;
    v6 = a1;
    v16 = &v30[8 * (unsigned int)v31];
    a3 = v25;
  }
  else
  {
    v16 = v32;
    v17 = 0;
    v18 = v32;
  }
  if ( v13 > 0 )
  {
    v19 = 0;
    do
    {
      *(_QWORD *)&v16[v19] = *(_QWORD *)(v11 + 4 * v19 + 32);
      v19 += 8;
      --v15;
    }
    while ( v15 );
    v18 = v30;
    v17 = v31;
  }
  v20 = v14 + v17;
  v27 = a3;
  v21 = v20;
  LODWORD(v31) = v20;
  v22 = sub_BB5290(v6);
  v23 = sub_BB5CE0(v22, v18, v21, a2, v27, v27, a4, a5);
  if ( v30 != v32 )
    _libc_free(v30, v18);
  return v23;
}
