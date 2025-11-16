// Function: sub_AAFC20
// Address: 0xaafc20
//
__int64 __fastcall sub_AAFC20(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // ebx
  unsigned int v6; // edx
  __int64 v7; // rax
  unsigned int v8; // edx
  __int64 v9; // rax
  __int64 *v10; // rsi
  __int64 v11; // r8
  unsigned int v12; // edx
  __int64 v13; // rax
  unsigned int v14; // edx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 *v18; // rcx
  int v19; // eax
  bool v20; // cc
  __int64 *v21; // rax
  __int64 v22[2]; // [rsp+50h] [rbp-100h] BYREF
  __int64 v23; // [rsp+60h] [rbp-F0h] BYREF
  int v24; // [rsp+68h] [rbp-E8h]
  __int64 v25[2]; // [rsp+70h] [rbp-E0h] BYREF
  __int64 v26; // [rsp+80h] [rbp-D0h] BYREF
  int v27; // [rsp+88h] [rbp-C8h]
  __int64 v28; // [rsp+90h] [rbp-C0h] BYREF
  int v29; // [rsp+98h] [rbp-B8h]
  __int64 v30; // [rsp+A0h] [rbp-B0h] BYREF
  unsigned int v31; // [rsp+A8h] [rbp-A8h]
  __int64 v32; // [rsp+B0h] [rbp-A0h] BYREF
  unsigned int v33; // [rsp+B8h] [rbp-98h]
  __int64 v34; // [rsp+C0h] [rbp-90h] BYREF
  unsigned int v35; // [rsp+C8h] [rbp-88h]
  __int64 v36; // [rsp+D0h] [rbp-80h] BYREF
  unsigned int v37; // [rsp+D8h] [rbp-78h]
  __int64 v38; // [rsp+E0h] [rbp-70h] BYREF
  __int64 *v39; // [rsp+E8h] [rbp-68h]
  __int64 v40; // [rsp+F0h] [rbp-60h] BYREF
  unsigned int v41; // [rsp+F8h] [rbp-58h]
  __int64 v42; // [rsp+100h] [rbp-50h] BYREF
  unsigned int v43; // [rsp+108h] [rbp-48h]
  __int64 v44; // [rsp+110h] [rbp-40h] BYREF
  int v45; // [rsp+118h] [rbp-38h]

  v4 = *(_DWORD *)(a2 + 8);
  if ( sub_AAF760(a2) || sub_AAF760(a3) || sub_AAFBB0(a2) || sub_AAFBB0(a3) )
  {
    *(_DWORD *)(a1 + 8) = v4;
    if ( v4 > 0x40 )
      sub_C43690(a1, 0, 0);
    else
      *(_QWORD *)a1 = 0;
  }
  else
  {
    sub_9865C0((__int64)v22, a2);
    sub_9865C0((__int64)&v44, a2 + 16);
    sub_C46F20(&v44, 1);
    v24 = v45;
    v23 = v44;
    sub_9865C0((__int64)v25, a3);
    sub_9865C0((__int64)&v44, a3 + 16);
    sub_C46F20(&v44, 1);
    v27 = v45;
    v26 = v44;
    sub_9865C0((__int64)&v40, (__int64)v22);
    v6 = v41;
    if ( v41 > 0x40 )
    {
      sub_C43C10(&v40, v25);
      v6 = v41;
      v7 = v40;
    }
    else
    {
      v7 = v25[0] ^ v40;
      v40 ^= v25[0];
    }
    v42 = v7;
    v43 = v6;
    v41 = 0;
    sub_9865C0((__int64)&v34, (__int64)v25);
    v8 = v35;
    if ( v35 > 0x40 )
    {
      sub_C43C10(&v34, &v26);
      v8 = v35;
      v9 = v34;
    }
    else
    {
      v9 = v26 ^ v34;
      v34 ^= v26;
    }
    v36 = v9;
    v10 = v22;
    v37 = v8;
    v35 = 0;
    sub_9865C0((__int64)&v30, (__int64)v22);
    v12 = v31;
    if ( v31 > 0x40 )
    {
      v10 = &v23;
      sub_C43C10(&v30, &v23);
      v12 = v31;
      v13 = v30;
    }
    else
    {
      v13 = v23 ^ v30;
      v30 ^= v23;
    }
    v33 = v12;
    v14 = v37;
    v32 = v13;
    v31 = 0;
    if ( v37 > 0x40 )
    {
      v10 = &v32;
      sub_C43BD0(&v36, &v32);
      v14 = v37;
      v15 = v36;
    }
    else
    {
      v15 = v36 | v13;
      v36 = v15;
    }
    LODWORD(v39) = v14;
    v16 = v43;
    v38 = v15;
    v37 = 0;
    if ( v43 > 0x40 )
    {
      v10 = &v38;
      sub_C43BD0(&v42, &v38);
      v16 = v43;
      v17 = v42;
    }
    else
    {
      v17 = v42 | v15;
      v18 = &v42;
      v42 = v17;
    }
    v45 = v16;
    v44 = v17;
    v43 = 0;
    sub_987160((__int64)&v44, (__int64)v10, v16, (__int64)v18, v11);
    v29 = v45;
    v28 = v44;
    sub_969240(&v38);
    sub_969240(&v32);
    sub_969240(&v30);
    sub_969240(&v36);
    sub_969240(&v34);
    sub_969240(&v42);
    sub_969240(&v40);
    v19 = sub_9871D0((__int64)&v28);
    sub_AAD9D0((__int64)&v28, v4 - v19);
    LODWORD(v38) = v4;
    v39 = &v28;
    sub_9865C0((__int64)&v44, (__int64)v22);
    sub_AAD630((__int64)&v40, (int *)&v38, (__int64)&v44, v25, &v26);
    sub_969240(&v44);
    sub_9865C0((__int64)&v44, (__int64)v25);
    sub_AAD630((__int64)&v42, (int *)&v38, (__int64)&v44, v22, &v23);
    sub_969240(&v44);
    v20 = (int)sub_C49970(&v40, &v42) <= 0;
    v21 = &v42;
    if ( !v20 )
      v21 = &v40;
    sub_9865C0(a1, (__int64)v21);
    sub_969240(&v42);
    sub_969240(&v40);
    sub_969240(&v28);
    sub_969240(&v26);
    sub_969240(v25);
    sub_969240(&v23);
    sub_969240(v22);
  }
  return a1;
}
