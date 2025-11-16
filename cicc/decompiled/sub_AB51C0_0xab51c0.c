// Function: sub_AB51C0
// Address: 0xab51c0
//
__int64 __fastcall sub_AB51C0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // eax
  unsigned int v6; // r9d
  __int64 v7; // rdx
  unsigned int v8; // r8d
  __int64 v9; // rax
  unsigned int v10; // eax
  int v11; // eax
  unsigned int v12; // [rsp+0h] [rbp-B0h]
  unsigned int v13; // [rsp+4h] [rbp-ACh]
  __int64 v14; // [rsp+8h] [rbp-A8h]
  __int64 v15; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v16; // [rsp+28h] [rbp-88h]
  __int64 v17; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v18; // [rsp+38h] [rbp-78h]
  __int64 v19; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v20; // [rsp+48h] [rbp-68h]
  __int64 v21; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v22; // [rsp+58h] [rbp-58h]
  __int64 v23; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v24; // [rsp+68h] [rbp-48h]
  __int64 v25; // [rsp+70h] [rbp-40h] BYREF
  int v26; // [rsp+78h] [rbp-38h]

  if ( !sub_AAF7D0(a2) && !sub_AAF7D0(a3) )
  {
    if ( sub_AAF760(a2) || sub_AAF760(a3) )
    {
      sub_AADB10(a1, *(_DWORD *)(a2 + 8), 1);
      return a1;
    }
    sub_9865C0((__int64)&v21, a2);
    sub_C46B40(&v21, a3 + 16);
    v5 = v22;
    v22 = 0;
    v24 = v5;
    v23 = v21;
    sub_C46A40(&v23, 1);
    v16 = v24;
    v15 = v23;
    sub_969240(&v21);
    sub_9865C0((__int64)&v23, a2 + 16);
    sub_C46B40(&v23, a3);
    v6 = v24;
    v7 = v23;
    v8 = v16;
    v18 = v24;
    v17 = v23;
    if ( v16 <= 0x40 )
    {
      v9 = v15;
      if ( v23 != v15 )
      {
LABEL_11:
        v20 = v8;
        v21 = v7;
        v22 = v6;
        v19 = v9;
        v18 = 0;
        v16 = 0;
        sub_AADC30((__int64)&v23, (__int64)&v19, &v21);
        sub_969240(&v19);
        sub_969240(&v21);
        if ( sub_AB01D0((__int64)&v23, a2) || sub_AB01D0((__int64)&v23, a3) )
        {
          sub_AADB10(a1, *(_DWORD *)(a2 + 8), 1);
        }
        else
        {
          v10 = v24;
          v24 = 0;
          *(_DWORD *)(a1 + 8) = v10;
          *(_QWORD *)a1 = v23;
          v11 = v26;
          v26 = 0;
          *(_DWORD *)(a1 + 24) = v11;
          *(_QWORD *)(a1 + 16) = v25;
        }
        sub_969240(&v25);
        sub_969240(&v23);
        goto LABEL_15;
      }
    }
    else
    {
      v12 = v16;
      v13 = v24;
      v14 = v23;
      if ( !(unsigned __int8)sub_C43C50(&v15, &v17) )
      {
        v9 = v15;
        v7 = v14;
        v6 = v13;
        v8 = v12;
        goto LABEL_11;
      }
    }
    sub_AADB10(a1, *(_DWORD *)(a2 + 8), 1);
LABEL_15:
    sub_969240(&v17);
    sub_969240(&v15);
    return a1;
  }
  sub_AADB10(a1, *(_DWORD *)(a2 + 8), 0);
  return a1;
}
