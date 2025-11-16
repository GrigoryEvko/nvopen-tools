// Function: sub_AB3510
// Address: 0xab3510
//
__int64 __fastcall sub_AB3510(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v6; // r12
  unsigned int v7; // eax
  unsigned int v8; // eax
  unsigned int v10; // eax
  unsigned int v11; // eax
  __int64 v12; // rbx
  __int64 v13; // rsi
  int v14; // eax
  int v15; // eax
  __int64 v16; // r14
  int v17; // eax
  int v18; // eax
  __int64 v19; // [rsp+18h] [rbp-B8h]
  __int64 v20; // [rsp+20h] [rbp-B0h] BYREF
  int v21; // [rsp+28h] [rbp-A8h]
  __int64 v22; // [rsp+30h] [rbp-A0h] BYREF
  int v23; // [rsp+38h] [rbp-98h]
  __int64 v24; // [rsp+40h] [rbp-90h] BYREF
  int v25; // [rsp+48h] [rbp-88h]
  __int64 v26; // [rsp+50h] [rbp-80h] BYREF
  int v27; // [rsp+58h] [rbp-78h]
  __int64 v28; // [rsp+60h] [rbp-70h] BYREF
  int v29; // [rsp+68h] [rbp-68h]
  __int64 v30; // [rsp+70h] [rbp-60h] BYREF
  __int64 v31; // [rsp+80h] [rbp-50h] BYREF
  int v32; // [rsp+88h] [rbp-48h]
  __int64 v33[8]; // [rsp+90h] [rbp-40h] BYREF

  v6 = a2;
  if ( sub_AAF760(a2) || sub_AAF7D0(a3) )
  {
    v7 = *(_DWORD *)(a2 + 8);
    *(_DWORD *)(a1 + 8) = v7;
    if ( v7 > 0x40 )
      sub_C43780(a1, a2);
    else
      *(_QWORD *)a1 = *(_QWORD *)a2;
    v8 = *(_DWORD *)(a2 + 24);
    *(_DWORD *)(a1 + 24) = v8;
    if ( v8 > 0x40 )
      sub_C43780(a1 + 16, a2 + 16);
    else
      *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
    return a1;
  }
  if ( sub_AAF760(a3) || sub_AAF7D0(a2) )
  {
    v10 = *(_DWORD *)(a3 + 8);
    *(_DWORD *)(a1 + 8) = v10;
    if ( v10 > 0x40 )
      sub_C43780(a1, a3);
    else
      *(_QWORD *)a1 = *(_QWORD *)a3;
    v11 = *(_DWORD *)(a3 + 24);
    *(_DWORD *)(a1 + 24) = v11;
    if ( v11 > 0x40 )
      sub_C43780(a1 + 16, a3 + 16);
    else
      *(_QWORD *)(a1 + 16) = *(_QWORD *)(a3 + 16);
    return a1;
  }
  if ( !sub_AB0100(a2) && sub_AB0100(a3) )
  {
    sub_AB3510(a1, a3, a2, a4);
    return a1;
  }
  v12 = a2 + 16;
  if ( !sub_AB0100(a2) && !sub_AB0100(a3) )
  {
    v19 = a3 + 16;
    if ( (int)sub_C49970(a3 + 16, a2) >= 0 && (int)sub_C49970(a2 + 16, a3) >= 0 )
    {
      v13 = a3;
      if ( (int)sub_C49970(a3, v6) >= 0 )
        v13 = v6;
      sub_9865C0((__int64)&v20, v13);
      sub_9865C0((__int64)&v24, v19);
      sub_C46F20(&v24, 1);
      v14 = v25;
      v25 = 0;
      v27 = v14;
      v26 = v24;
      sub_9865C0((__int64)&v28, v6 + 16);
      sub_C46F20(&v28, 1);
      v32 = v29;
      v31 = v28;
      v29 = 0;
      if ( (int)sub_C49970(&v26, &v31) > 0 )
        v12 = a3 + 16;
      sub_9865C0((__int64)&v22, v12);
      sub_969240(&v31);
      sub_969240(&v28);
      sub_969240(&v26);
      sub_969240(&v24);
      if ( sub_9867B0((__int64)&v20) && sub_9867B0((__int64)&v22) )
      {
        sub_AADB10(a1, *(_DWORD *)(v6 + 8), 1);
      }
      else
      {
        v32 = v23;
        v23 = 0;
        v31 = v22;
        v15 = v21;
        v21 = 0;
        v29 = v15;
        v28 = v20;
        sub_AADC30(a1, (__int64)&v28, &v31);
        sub_969240(&v28);
        sub_969240(&v31);
      }
      sub_969240(&v22);
      sub_969240(&v20);
      return a1;
    }
LABEL_46:
    sub_9865C0((__int64)&v26, a2 + 16);
    sub_9865C0((__int64)&v24, a3);
    sub_AADC30((__int64)&v31, (__int64)&v24, &v26);
    sub_9865C0((__int64)&v22, v19);
    sub_9865C0((__int64)&v20, a2);
    sub_AADC30((__int64)&v28, (__int64)&v20, &v22);
    sub_AB0360(a1, (__int64)&v28, (__int64)&v31, a4);
    sub_969240(&v30);
    sub_969240(&v28);
    sub_969240(&v20);
    sub_969240(&v22);
    sub_969240(v33);
    sub_969240(&v31);
    sub_969240(&v24);
    sub_969240(&v26);
    return a1;
  }
  if ( !sub_AB0100(a3) )
  {
    v19 = a3 + 16;
    if ( (int)sub_C49970(a3 + 16, a2 + 16) <= 0 || (int)sub_C49970(a3, a2) >= 0 )
    {
      sub_AAF450(a1, a2);
      return a1;
    }
    if ( (int)sub_C49970(a3, a2 + 16) > 0 || (int)sub_C49970(a2, v19) > 0 )
    {
      if ( (int)sub_C49970(a2 + 16, a3) < 0 )
      {
        if ( (int)sub_C49970(v19, a2) < 0 )
          goto LABEL_46;
        if ( (int)sub_C49970(a2, v19) <= 0 )
        {
          sub_9865C0((__int64)&v31, v12);
          sub_9865C0((__int64)&v28, a3);
          sub_AADC30(a1, (__int64)&v28, &v31);
          sub_969240(&v28);
          sub_969240(&v31);
          return a1;
        }
      }
      sub_9865C0((__int64)&v31, v19);
      sub_9865C0((__int64)&v28, a2);
      sub_AADC30(a1, (__int64)&v28, &v31);
      sub_969240(&v28);
      sub_969240(&v31);
      return a1;
    }
LABEL_44:
    sub_AADB10(a1, *(_DWORD *)(a2 + 8), 1);
    return a1;
  }
  if ( (int)sub_C49970(a3, a2 + 16) <= 0 )
    goto LABEL_44;
  v16 = a3 + 16;
  if ( (int)sub_C49970(a2, a3 + 16) <= 0 )
    goto LABEL_44;
  if ( (int)sub_C49970(a3, a2) < 0 )
    a2 = a3;
  sub_9865C0((__int64)&v24, a2);
  if ( (int)sub_C49970(v16, v12) <= 0 )
    v16 = v12;
  sub_9865C0((__int64)&v26, v16);
  v17 = v27;
  v27 = 0;
  v32 = v17;
  v31 = v26;
  v18 = v25;
  v25 = 0;
  v29 = v18;
  v28 = v24;
  sub_AADC30(a1, (__int64)&v28, &v31);
  sub_969240(&v28);
  sub_969240(&v31);
  sub_969240(&v26);
  sub_969240(&v24);
  return a1;
}
