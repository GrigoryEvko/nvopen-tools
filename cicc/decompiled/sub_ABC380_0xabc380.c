// Function: sub_ABC380
// Address: 0xabc380
//
__int64 __fastcall sub_ABC380(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 *v6; // rax
  __int64 *v7; // r14
  unsigned int v8; // ebx
  __int64 *v9; // rsi
  __int64 v10; // rdx
  __int64 *v11; // rsi
  int v12; // eax
  unsigned int v13; // esi
  __int64 v14; // rdx
  __int64 v15; // r8
  __int64 v16; // rdx
  __int64 v17; // rcx
  int v18; // eax
  __int64 *v19; // rsi
  int v20; // eax
  __int64 *v21; // rsi
  int v22; // eax
  int v23; // eax
  __int64 v24; // rdx
  __int64 v25; // rcx
  int v26; // eax
  __int64 *v27; // rsi
  int v28; // [rsp+8h] [rbp-108h]
  __int64 v29[2]; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v30[2]; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v31; // [rsp+50h] [rbp-C0h] BYREF
  unsigned int v32; // [rsp+58h] [rbp-B8h]
  __int64 v33; // [rsp+60h] [rbp-B0h] BYREF
  int v34; // [rsp+68h] [rbp-A8h]
  __int64 v35; // [rsp+70h] [rbp-A0h] BYREF
  int v36; // [rsp+78h] [rbp-98h]
  __int64 v37; // [rsp+80h] [rbp-90h] BYREF
  int v38; // [rsp+88h] [rbp-88h]
  __int64 v39; // [rsp+90h] [rbp-80h] BYREF
  int v40; // [rsp+98h] [rbp-78h]
  __int64 v41; // [rsp+A0h] [rbp-70h] BYREF
  int v42; // [rsp+A8h] [rbp-68h]
  __int64 v43; // [rsp+B0h] [rbp-60h] BYREF
  int v44; // [rsp+B8h] [rbp-58h]
  __int64 v45[2]; // [rsp+C0h] [rbp-50h] BYREF
  __int64 v46[8]; // [rsp+D0h] [rbp-40h] BYREF

  if ( sub_AAF7D0(a2) || sub_AAF7D0((__int64)a3) )
    goto LABEL_3;
  v6 = sub_9876C0(a3);
  v7 = v6;
  if ( !v6 )
  {
LABEL_10:
    sub_ABBBB0((__int64)v45, (__int64)a3, 0);
    sub_AB0A00((__int64)v29, (__int64)v45);
    sub_AB0910((__int64)v30, (__int64)v45);
    if ( sub_9867B0((__int64)v30) )
    {
      sub_AADB10(a1, *(_DWORD *)(a2 + 8), 0);
      goto LABEL_21;
    }
    if ( sub_9867B0((__int64)v29) )
      sub_C46250(v29);
    sub_AB14C0((__int64)&v31, a2);
    sub_AB13A0((__int64)&v33, a2);
    if ( v32 > 0x40 )
      v10 = *(_QWORD *)(v31 + 8LL * ((v32 - 1) >> 6));
    else
      v10 = v31;
    if ( (v10 & (1LL << ((unsigned __int8)v32 - 1))) != 0 )
    {
      if ( !sub_986C60(&v33, v34 - 1) )
      {
        sub_9865C0((__int64)&v39, (__int64)v30);
        sub_AADAA0((__int64)&v41, (__int64)&v39, v16, v17, (__int64)&v39);
        sub_C46A40(&v41, 1);
        v18 = v42;
        v42 = 0;
        v44 = v18;
        v43 = v41;
        v19 = &v43;
        if ( (int)sub_C49970(&v31, &v43) > 0 )
          v19 = &v31;
        sub_9865C0((__int64)&v35, (__int64)v19);
        sub_969240(&v43);
        sub_969240(&v41);
        sub_969240(&v39);
        sub_9865C0((__int64)&v39, (__int64)v30);
        sub_C46F20(&v39, 1);
        v20 = v40;
        v40 = 0;
        v42 = v20;
        v41 = v39;
        v21 = &v41;
        if ( (int)sub_C49970(&v33, &v41) < 0 )
          v21 = &v33;
        sub_9865C0((__int64)&v43, (__int64)v21);
        sub_C46A40(&v43, 1);
        v38 = v44;
        v37 = v43;
        sub_969240(&v41);
        sub_969240(&v39);
        v22 = v38;
        v38 = 0;
        v44 = v22;
        v43 = v37;
        v23 = v36;
        v36 = 0;
        v42 = v23;
        v41 = v35;
        sub_AADC30(a1, (__int64)&v41, &v43);
        sub_969240(&v41);
        sub_969240(&v43);
        sub_969240(&v37);
        sub_969240(&v35);
        goto LABEL_20;
      }
      sub_9865C0((__int64)&v41, (__int64)v29);
      sub_AADAA0((__int64)&v43, (__int64)&v41, v14, (__int64)&v43, v15);
      v28 = sub_C49970(&v31, &v43);
      sub_969240(&v43);
      sub_969240(&v41);
      if ( v28 <= 0 )
      {
        sub_9865C0((__int64)&v39, (__int64)v30);
        sub_AADAA0((__int64)&v41, (__int64)&v39, v24, v25, (__int64)&v39);
        sub_C46A40(&v41, 1);
        v26 = v42;
        v42 = 0;
        v44 = v26;
        v43 = v41;
        v27 = &v43;
        if ( (int)sub_C49970(&v31, &v43) > 0 )
          v27 = &v31;
        sub_9865C0((__int64)&v37, (__int64)v27);
        sub_969240(&v43);
        sub_969240(&v41);
        sub_969240(&v39);
        sub_9691E0((__int64)&v41, *(_DWORD *)(a2 + 8), 1, 0, 0);
        v44 = v38;
        v38 = 0;
        v43 = v37;
        sub_AADC30(a1, (__int64)&v43, &v41);
        sub_969240(&v43);
        sub_969240(&v41);
        sub_969240(&v37);
        goto LABEL_20;
      }
    }
    else if ( (int)sub_C49970(&v33, v29) >= 0 )
    {
      sub_9865C0((__int64)&v39, (__int64)v30);
      sub_C46F20(&v39, 1);
      v42 = v40;
      v40 = 0;
      v41 = v39;
      v11 = &v41;
      if ( (int)sub_C49970(&v33, &v41) < 0 )
        v11 = &v33;
      sub_9865C0((__int64)&v43, (__int64)v11);
      sub_C46A40(&v43, 1);
      v38 = v44;
      v37 = v43;
      sub_969240(&v41);
      sub_969240(&v39);
      v12 = v38;
      v13 = *(_DWORD *)(a2 + 8);
      v38 = 0;
      v44 = v12;
      v43 = v37;
      sub_9691E0((__int64)&v41, v13, 0, 0, 0);
      sub_AADC30(a1, (__int64)&v41, &v43);
      sub_969240(&v41);
      sub_969240(&v43);
      sub_969240(&v37);
      goto LABEL_20;
    }
    sub_AAF450(a1, a2);
LABEL_20:
    sub_969240(&v33);
    sub_969240(&v31);
LABEL_21:
    sub_969240(v30);
    sub_969240(v29);
    sub_969240(v46);
    sub_969240(v45);
    return a1;
  }
  v8 = *((_DWORD *)v6 + 2);
  if ( v8 <= 0x40 )
  {
    if ( *v6 )
    {
LABEL_8:
      v9 = sub_9876C0((__int64 *)a2);
      if ( v9 )
      {
        sub_C4B8A0(v45, v9, v7);
        sub_AADBC0(a1, v45);
        sub_969240(v45);
        return a1;
      }
      goto LABEL_10;
    }
  }
  else if ( v8 != (unsigned int)sub_C444A0(v6) )
  {
    goto LABEL_8;
  }
LABEL_3:
  sub_AADB10(a1, *(_DWORD *)(a2 + 8), 0);
  return a1;
}
