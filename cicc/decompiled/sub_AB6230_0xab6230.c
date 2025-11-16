// Function: sub_AB6230
// Address: 0xab6230
//
__int64 __fastcall sub_AB6230(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v6; // rsi
  __int64 *v7; // rsi
  unsigned int v8; // eax
  unsigned int v9; // eax
  unsigned int v10; // eax
  int v11; // eax
  __int64 v12; // [rsp+10h] [rbp-A0h] BYREF
  unsigned int v13; // [rsp+18h] [rbp-98h]
  __int64 v14; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v15; // [rsp+28h] [rbp-88h]
  __int64 v16; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v17; // [rsp+38h] [rbp-78h]
  __int64 v18; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v19; // [rsp+48h] [rbp-68h]
  __int64 v20; // [rsp+50h] [rbp-60h] BYREF
  int v21; // [rsp+58h] [rbp-58h]
  __int64 v22; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v23; // [rsp+68h] [rbp-48h]
  __int64 v24[8]; // [rsp+70h] [rbp-40h] BYREF

  if ( sub_AAF7D0(a2) || sub_AAF7D0(a3) )
  {
    sub_AADB10(a1, *(_DWORD *)(a2 + 8), 0);
  }
  else
  {
    sub_AB0A00((__int64)&v22, a3);
    sub_AB0A00((__int64)&v18, a2);
    v6 = &v22;
    if ( (int)sub_C49970(&v18, &v22) > 0 )
      v6 = &v18;
    v13 = *((_DWORD *)v6 + 2);
    if ( v13 > 0x40 )
      sub_C43780(&v12, v6);
    else
      v12 = *v6;
    if ( v19 > 0x40 && v18 )
      j_j___libc_free_0_0(v18);
    sub_969240(&v22);
    sub_AB0910((__int64)&v18, a3);
    sub_AB0910((__int64)&v16, a2);
    v7 = &v16;
    if ( (int)sub_C49970(&v16, &v18) <= 0 )
      v7 = &v18;
    v23 = *((_DWORD *)v7 + 2);
    if ( v23 > 0x40 )
      sub_C43780(&v22, v7);
    else
      v22 = *v7;
    sub_C46A40(&v22, 1);
    v15 = v23;
    v14 = v22;
    sub_969240(&v16);
    sub_969240(&v18);
    v8 = v15;
    v15 = 0;
    v23 = v8;
    v22 = v14;
    v9 = v13;
    v13 = 0;
    v17 = v9;
    v16 = v12;
    sub_9875E0((__int64)&v18, &v16, &v22);
    if ( v17 > 0x40 && v16 )
      j_j___libc_free_0_0(v16);
    sub_969240(&v22);
    if ( sub_AAFBB0(a2) || sub_AAFBB0(a3) )
    {
      sub_AB3510((__int64)&v22, a2, a3, 1u);
      sub_AB2160(a1, (__int64)&v18, (__int64)&v22, 1u);
      sub_969240(v24);
      sub_969240(&v22);
    }
    else
    {
      v10 = v19;
      v19 = 0;
      *(_DWORD *)(a1 + 8) = v10;
      *(_QWORD *)a1 = v18;
      v11 = v21;
      v21 = 0;
      *(_DWORD *)(a1 + 24) = v11;
      *(_QWORD *)(a1 + 16) = v20;
    }
    sub_969240(&v20);
    sub_969240(&v18);
    sub_969240(&v14);
    sub_969240(&v12);
  }
  return a1;
}
