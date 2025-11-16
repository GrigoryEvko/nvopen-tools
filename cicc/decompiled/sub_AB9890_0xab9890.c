// Function: sub_AB9890
// Address: 0xab9890
//
__int64 __fastcall sub_AB9890(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdx
  unsigned int v6; // eax
  __int64 v7; // rdx
  __int64 v8; // [rsp+18h] [rbp-C8h]
  bool v9; // [rsp+18h] [rbp-C8h]
  __int64 v10; // [rsp+30h] [rbp-B0h] BYREF
  unsigned int v11; // [rsp+38h] [rbp-A8h]
  __int64 v12; // [rsp+40h] [rbp-A0h] BYREF
  unsigned int v13; // [rsp+48h] [rbp-98h]
  __int64 v14; // [rsp+50h] [rbp-90h] BYREF
  unsigned int v15; // [rsp+58h] [rbp-88h]
  __int64 v16; // [rsp+60h] [rbp-80h] BYREF
  unsigned int v17; // [rsp+68h] [rbp-78h]
  __int64 v18; // [rsp+70h] [rbp-70h] BYREF
  unsigned int v19; // [rsp+78h] [rbp-68h]
  __int64 v20; // [rsp+80h] [rbp-60h] BYREF
  unsigned int v21; // [rsp+88h] [rbp-58h]
  __int64 v22; // [rsp+90h] [rbp-50h] BYREF
  unsigned int v23; // [rsp+98h] [rbp-48h]
  __int64 v24; // [rsp+A0h] [rbp-40h] BYREF
  unsigned int v25; // [rsp+A8h] [rbp-38h]

  if ( !sub_AAF7D0(a2) && !sub_AAF7D0(a3) )
  {
    sub_AB13A0((__int64)&v20, a2);
    sub_AB0A00((__int64)&v22, a3);
    sub_9865C0((__int64)&v24, (__int64)&v20);
    sub_C44D10(&v24, &v22);
    sub_C46A40(&v24, 1);
    v11 = v25;
    v10 = v24;
    sub_969240(&v22);
    sub_969240(&v20);
    sub_AB14C0((__int64)&v22, a2);
    sub_AB0910((__int64)&v24, a3);
    sub_9865C0((__int64)&v12, (__int64)&v22);
    sub_C44D10(&v12, &v24);
    sub_969240(&v24);
    sub_969240(&v22);
    sub_AB13A0((__int64)&v20, a2);
    sub_AB0910((__int64)&v22, a3);
    sub_9865C0((__int64)&v24, (__int64)&v20);
    sub_C44D10(&v24, &v22);
    sub_C46A40(&v24, 1);
    v15 = v25;
    v14 = v24;
    sub_969240(&v22);
    sub_969240(&v20);
    sub_AB14C0((__int64)&v22, a2);
    sub_AB0A00((__int64)&v24, a3);
    sub_9865C0((__int64)&v16, (__int64)&v22);
    sub_C44D10(&v16, &v24);
    sub_969240(&v24);
    sub_969240(&v22);
    v19 = 1;
    v18 = 0;
    v21 = 1;
    v20 = 0;
    sub_AB14C0((__int64)&v24, a2);
    if ( v25 > 0x40 )
      v5 = *(_QWORD *)(v24 + 8LL * ((v25 - 1) >> 6));
    else
      v5 = v24;
    v8 = v5 & (1LL << ((unsigned __int8)v25 - 1));
    sub_969240(&v24);
    if ( v8 )
    {
      sub_AB13A0((__int64)&v24, a2);
      v9 = sub_986C60(&v24, v25 - 1);
      sub_969240(&v24);
      if ( v9 )
      {
        if ( v21 <= 0x40 && v17 <= 0x40 )
        {
          v21 = v17;
          v20 = v16;
        }
        else
        {
          sub_C43990(&v20, &v16);
        }
        if ( v19 <= 0x40 && (v6 = v15, v15 <= 0x40) )
        {
          v7 = v14;
          v18 = v14;
        }
        else
        {
          sub_C43990(&v18, &v14);
          v6 = v19;
          v7 = v18;
        }
        goto LABEL_14;
      }
      if ( v21 <= 0x40 && v17 <= 0x40 )
      {
        v21 = v17;
        v20 = v16;
      }
      else
      {
        sub_C43990(&v20, &v16);
      }
      if ( v19 > 0x40 )
        goto LABEL_13;
      v6 = v11;
      if ( v11 > 0x40 )
        goto LABEL_13;
    }
    else
    {
      if ( v21 <= 0x40 && v13 <= 0x40 )
      {
        v21 = v13;
        v20 = v12;
      }
      else
      {
        sub_C43990(&v20, &v12);
      }
      if ( v19 > 0x40 || (v6 = v11, v11 > 0x40) )
      {
LABEL_13:
        sub_C43990(&v18, &v10);
        v6 = v19;
        v7 = v18;
LABEL_14:
        v25 = v6;
        v24 = v7;
        v23 = v21;
        v22 = v20;
        v19 = 0;
        v21 = 0;
        sub_9875E0(a1, &v22, &v24);
        sub_969240(&v22);
        sub_969240(&v24);
        sub_969240(&v20);
        sub_969240(&v18);
        sub_969240(&v16);
        sub_969240(&v14);
        sub_969240(&v12);
        sub_969240(&v10);
        return a1;
      }
    }
    v7 = v10;
    v18 = v10;
    goto LABEL_14;
  }
  sub_AADB10(a1, *(_DWORD *)(a2 + 8), 0);
  return a1;
}
