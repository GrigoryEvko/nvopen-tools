// Function: sub_AB4F10
// Address: 0xab4f10
//
__int64 __fastcall sub_AB4F10(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v6; // eax
  unsigned int v7; // edx
  __int64 v8; // rax
  __int64 v9; // rsi
  unsigned int v10; // eax
  int v11; // eax
  unsigned int v12; // [rsp+8h] [rbp-A8h]
  __int64 v13; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v14; // [rsp+28h] [rbp-88h]
  __int64 v15; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v16; // [rsp+38h] [rbp-78h]
  __int64 v17; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v18; // [rsp+48h] [rbp-68h]
  __int64 v19; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v20; // [rsp+58h] [rbp-58h]
  __int64 v21; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v22; // [rsp+68h] [rbp-48h]
  __int64 v23; // [rsp+70h] [rbp-40h] BYREF
  int v24; // [rsp+78h] [rbp-38h]

  if ( !sub_AAF7D0(a2) && !sub_AAF7D0(a3) )
  {
    if ( sub_AAF760(a2) || sub_AAF760(a3) )
    {
      sub_AADB10(a1, *(_DWORD *)(a2 + 8), 1);
      return a1;
    }
    sub_9865C0((__int64)&v21, a2);
    sub_C45EE0(&v21, a3);
    v14 = v22;
    v13 = v21;
    sub_9865C0((__int64)&v19, a2 + 16);
    sub_C45EE0(&v19, a3 + 16);
    v6 = v20;
    v20 = 0;
    v22 = v6;
    v21 = v19;
    sub_C46F20(&v21, 1);
    v16 = v22;
    v15 = v21;
    sub_969240(&v19);
    v7 = v14;
    if ( v14 <= 0x40 )
    {
      v8 = v13;
      v9 = v15;
      if ( v13 != v15 )
      {
LABEL_11:
        v19 = v9;
        v18 = v7;
        v20 = v16;
        v17 = v8;
        v16 = 0;
        v14 = 0;
        sub_AADC30((__int64)&v21, (__int64)&v17, &v19);
        sub_969240(&v17);
        sub_969240(&v19);
        if ( sub_AB01D0((__int64)&v21, a2) || sub_AB01D0((__int64)&v21, a3) )
        {
          sub_AADB10(a1, *(_DWORD *)(a2 + 8), 1);
        }
        else
        {
          v10 = v22;
          v22 = 0;
          *(_DWORD *)(a1 + 8) = v10;
          *(_QWORD *)a1 = v21;
          v11 = v24;
          v24 = 0;
          *(_DWORD *)(a1 + 24) = v11;
          *(_QWORD *)(a1 + 16) = v23;
        }
        sub_969240(&v23);
        sub_969240(&v21);
        goto LABEL_15;
      }
    }
    else
    {
      v12 = v14;
      if ( !(unsigned __int8)sub_C43C50(&v13, &v15) )
      {
        v8 = v13;
        v9 = v15;
        v7 = v12;
        goto LABEL_11;
      }
    }
    sub_AADB10(a1, *(_DWORD *)(a2 + 8), 1);
LABEL_15:
    sub_969240(&v15);
    sub_969240(&v13);
    return a1;
  }
  sub_AADB10(a1, *(_DWORD *)(a2 + 8), 0);
  return a1;
}
