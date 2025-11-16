// Function: sub_AB64F0
// Address: 0xab64f0
//
__int64 __fastcall sub_AB64F0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v6; // rsi
  __int64 *v7; // rsi
  unsigned int v8; // eax
  unsigned int v9; // eax
  int v10; // eax
  __int64 v11; // [rsp+8h] [rbp-A8h]
  unsigned int v12; // [rsp+14h] [rbp-9Ch]
  __int64 v13; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v14; // [rsp+28h] [rbp-88h]
  __int64 v15; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v16; // [rsp+38h] [rbp-78h]
  __int64 v17; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v18; // [rsp+48h] [rbp-68h]
  __int64 v19; // [rsp+50h] [rbp-60h] BYREF
  int v20; // [rsp+58h] [rbp-58h]
  __int64 v21; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v22; // [rsp+68h] [rbp-48h]
  __int64 v23[8]; // [rsp+70h] [rbp-40h] BYREF

  if ( sub_AAF7D0(a2) || sub_AAF7D0(a3) )
  {
    sub_AADB10(a1, *(_DWORD *)(a2 + 8), 0);
  }
  else
  {
    sub_AB14C0((__int64)&v21, a3);
    sub_AB14C0((__int64)&v17, a2);
    v6 = &v21;
    if ( (int)sub_C4C880(&v17, &v21) < 0 )
      v6 = &v17;
    v14 = *((_DWORD *)v6 + 2);
    if ( v14 > 0x40 )
      sub_C43780(&v13, v6);
    else
      v13 = *v6;
    sub_969240(&v17);
    sub_969240(&v21);
    sub_AB13A0((__int64)&v17, a3);
    sub_AB13A0((__int64)&v15, a2);
    v7 = &v15;
    if ( (int)sub_C4C880(&v15, &v17) >= 0 )
      v7 = &v17;
    v22 = *((_DWORD *)v7 + 2);
    if ( v22 > 0x40 )
      sub_C43780(&v21, v7);
    else
      v21 = *v7;
    sub_C46A40(&v21, 1);
    v12 = v22;
    v11 = v21;
    sub_969240(&v15);
    if ( v18 > 0x40 && v17 )
      j_j___libc_free_0_0(v17);
    v22 = v12;
    v21 = v11;
    v8 = v14;
    v14 = 0;
    v16 = v8;
    v15 = v13;
    sub_9875E0((__int64)&v17, &v15, &v21);
    sub_969240(&v15);
    sub_969240(&v21);
    if ( sub_AB0120(a2) || sub_AB0120(a3) )
    {
      sub_AB3510((__int64)&v21, a2, a3, 2u);
      sub_AB2160(a1, (__int64)&v17, (__int64)&v21, 2u);
      sub_969240(v23);
      sub_969240(&v21);
    }
    else
    {
      v9 = v18;
      v18 = 0;
      *(_DWORD *)(a1 + 8) = v9;
      *(_QWORD *)a1 = v17;
      v10 = v20;
      v20 = 0;
      *(_DWORD *)(a1 + 24) = v10;
      *(_QWORD *)(a1 + 16) = v19;
    }
    sub_969240(&v19);
    sub_969240(&v17);
    sub_969240(&v13);
  }
  return a1;
}
