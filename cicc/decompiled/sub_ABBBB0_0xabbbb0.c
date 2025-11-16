// Function: sub_ABBBB0
// Address: 0xabbbb0
//
__int64 __fastcall sub_ABBBB0(__int64 a1, __int64 a2, char a3)
{
  unsigned int v6; // r15d
  __int64 v7; // rdx
  __int64 v8; // rax
  bool v9; // al
  unsigned int v10; // r15d
  __int64 v11; // rdx
  __int64 v12; // rax
  unsigned int v13; // eax
  unsigned int v14; // esi
  unsigned int v15; // eax
  __int64 v16; // rdx
  __int64 v17; // rcx
  unsigned int v18; // r8d
  __int64 v19; // rdx
  __int64 v20; // rcx
  unsigned int v21; // r8d
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // r8
  unsigned __int64 v25; // rax
  unsigned int v26; // eax
  __int64 v27; // rcx
  unsigned __int64 v28; // rax
  unsigned int v29; // eax
  __int64 v30; // rsi
  __int64 v31; // rcx
  __int64 v32; // r8
  unsigned int v33; // eax
  __int64 *v34; // rsi
  __int64 v35; // rcx
  __int64 v36; // r8
  unsigned __int64 v37; // rax
  __int64 *v38; // rsi
  unsigned int v39; // esi
  unsigned __int64 v40; // rax
  unsigned int v41; // eax
  __int64 v42; // [rsp+20h] [rbp-A0h] BYREF
  unsigned int v43; // [rsp+28h] [rbp-98h]
  __int64 v44; // [rsp+30h] [rbp-90h] BYREF
  int v45; // [rsp+38h] [rbp-88h]
  unsigned __int64 v46; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v47; // [rsp+48h] [rbp-78h]
  unsigned __int64 v48; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v49; // [rsp+58h] [rbp-68h]
  unsigned __int64 v50; // [rsp+60h] [rbp-60h] BYREF
  unsigned int v51; // [rsp+68h] [rbp-58h]
  unsigned __int64 v52; // [rsp+70h] [rbp-50h] BYREF
  unsigned int v53; // [rsp+78h] [rbp-48h]
  unsigned __int64 v54; // [rsp+80h] [rbp-40h] BYREF
  unsigned int v55; // [rsp+88h] [rbp-38h]

  if ( sub_AAF7D0(a2) )
  {
    sub_AADB10(a1, *(_DWORD *)(a2 + 8), 0);
    return a1;
  }
  if ( !sub_AB0120(a2) )
  {
    sub_AB14C0((__int64)&v42, a2);
    sub_AB13A0((__int64)&v44, a2);
    if ( a3 && (unsigned __int8)sub_986B30(&v42, a2, v16, v17, v18) )
    {
      if ( (unsigned __int8)sub_986B30(&v44, a2, v19, v20, v21) )
      {
        sub_AADB10(a1, *(_DWORD *)(a2 + 8), 0);
LABEL_48:
        sub_969240(&v44);
        sub_969240(&v42);
        return a1;
      }
      sub_C46250(&v42);
    }
    v22 = v42;
    if ( v43 > 0x40 )
      v22 = *(_QWORD *)(v42 + 8LL * ((v43 - 1) >> 6));
    if ( (v22 & (1LL << ((unsigned __int8)v43 - 1))) != 0 )
    {
      if ( sub_986C60(&v44, v45 - 1) )
      {
        sub_9865C0((__int64)&v50, (__int64)&v42);
        if ( v51 > 0x40 )
        {
          sub_C43D10(&v50, &v42, v51, v23, v24);
        }
        else
        {
          v25 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v51) & ~v50;
          if ( !v51 )
            v25 = 0;
          v50 = v25;
        }
        sub_C46250(&v50);
        v53 = v51;
        v51 = 0;
        v52 = v50;
        sub_C46A40(&v52, 1);
        v26 = v53;
        v53 = 0;
        v55 = v26;
        v54 = v52;
        sub_9865C0((__int64)&v46, (__int64)&v44);
        if ( v47 > 0x40 )
        {
          sub_C43D10(&v46, &v44, v47, v27, &v52);
        }
        else
        {
          v28 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v47) & ~v46;
          if ( !v47 )
            v28 = 0;
          v46 = v28;
        }
        sub_C46250(&v46);
        v29 = v47;
        v47 = 0;
        v49 = v29;
        v48 = v46;
        sub_AADC30(a1, (__int64)&v48, (__int64 *)&v54);
        sub_969240((__int64 *)&v48);
        sub_969240((__int64 *)&v46);
        sub_969240((__int64 *)&v54);
        sub_969240((__int64 *)&v52);
        sub_969240((__int64 *)&v50);
      }
      else
      {
        sub_9865C0((__int64)&v48, (__int64)&v42);
        if ( v49 > 0x40 )
        {
          sub_C43D10(&v48, &v42, v49, v35, v36);
        }
        else
        {
          v37 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v49) & ~v48;
          if ( !v49 )
            v37 = 0;
          v48 = v37;
        }
        sub_C46250(&v48);
        v51 = v49;
        v49 = 0;
        v50 = v48;
        v38 = &v44;
        if ( (int)sub_C49970(&v50, &v44) > 0 )
          v38 = (__int64 *)&v50;
        sub_9865C0((__int64)&v52, (__int64)v38);
        sub_C46A40(&v52, 1);
        v39 = *(_DWORD *)(a2 + 8);
        v55 = v53;
        v53 = 0;
        v54 = v52;
        sub_9691E0((__int64)&v46, v39, 0, 0, 0);
        sub_9875E0(a1, (__int64 *)&v46, (__int64 *)&v54);
        sub_969240((__int64 *)&v46);
        sub_969240((__int64 *)&v54);
        sub_969240((__int64 *)&v52);
        sub_969240((__int64 *)&v50);
        sub_969240((__int64 *)&v48);
      }
    }
    else
    {
      sub_9865C0((__int64)&v50, (__int64)&v44);
      sub_C46A40(&v50, 1);
      v41 = v51;
      v51 = 0;
      v53 = v41;
      v52 = v50;
      sub_9865C0((__int64)&v54, (__int64)&v42);
      sub_AADC30(a1, (__int64)&v54, (__int64 *)&v52);
      sub_969240((__int64 *)&v54);
      sub_969240((__int64 *)&v52);
      sub_969240((__int64 *)&v50);
    }
    goto LABEL_48;
  }
  v6 = *(_DWORD *)(a2 + 24);
  v7 = *(_QWORD *)(a2 + 16);
  v49 = 1;
  v48 = 0;
  v8 = 1LL << ((unsigned __int8)v6 - 1);
  if ( v6 > 0x40 )
  {
    if ( (*(_QWORD *)(v7 + 8LL * ((v6 - 1) >> 6)) & v8) == 0 )
    {
      v9 = v6 == (unsigned int)sub_C444A0(a2 + 16);
      goto LABEL_8;
    }
LABEL_16:
    v10 = *(_DWORD *)(a2 + 8);
LABEL_9:
    v11 = *(_QWORD *)a2;
    v12 = 1LL << ((unsigned __int8)v10 - 1);
    if ( v10 > 0x40 )
    {
      if ( (*(_QWORD *)(v11 + 8LL * ((v10 - 1) >> 6)) & v12) != 0 || v10 == (unsigned int)sub_C444A0(a2) )
        goto LABEL_11;
    }
    else if ( (v12 & v11) != 0 || !v11 )
    {
      goto LABEL_11;
    }
    v30 = a2 + 16;
    sub_9865C0((__int64)&v50, a2 + 16);
    if ( v51 <= 0x40 )
    {
      v40 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v51) & ~v50;
      if ( !v51 )
        v40 = 0;
      v50 = v40;
    }
    else
    {
      sub_C43D10(&v50, v30, v51, v31, v32);
    }
    sub_C46250(&v50);
    v53 = v51;
    v51 = 0;
    v52 = v50;
    sub_C46A40(&v52, 1);
    v33 = v53;
    v53 = 0;
    v55 = v33;
    v54 = v52;
    v34 = (__int64 *)&v54;
    if ( (int)sub_C49970(a2, &v54) < 0 )
      v34 = (__int64 *)a2;
    sub_AAD590((__int64)&v48, (__int64)v34);
    sub_969240((__int64 *)&v54);
    sub_969240((__int64 *)&v52);
    sub_969240((__int64 *)&v50);
    goto LABEL_12;
  }
  if ( (v8 & v7) != 0 )
    goto LABEL_16;
  v9 = v7 == 0;
LABEL_8:
  v10 = *(_DWORD *)(a2 + 8);
  if ( v9 )
    goto LABEL_9;
LABEL_11:
  sub_9691E0((__int64)&v54, v10, 0, 0, 0);
  v48 = v54;
  v13 = v55;
  v55 = 0;
  v49 = v13;
  sub_969240((__int64 *)&v54);
LABEL_12:
  v14 = *(_DWORD *)(a2 + 8);
  if ( a3 )
  {
    sub_986680((__int64)&v52, v14);
    sub_9865C0((__int64)&v54, (__int64)&v48);
    sub_AADC30(a1, (__int64)&v54, (__int64 *)&v52);
    sub_969240((__int64 *)&v54);
    sub_969240((__int64 *)&v52);
  }
  else
  {
    sub_986680((__int64)&v50, v14);
    sub_C46A40(&v50, 1);
    v15 = v51;
    v51 = 0;
    v53 = v15;
    v52 = v50;
    sub_9865C0((__int64)&v54, (__int64)&v48);
    sub_AADC30(a1, (__int64)&v54, (__int64 *)&v52);
    sub_969240((__int64 *)&v54);
    sub_969240((__int64 *)&v52);
    sub_969240((__int64 *)&v50);
  }
  sub_969240((__int64 *)&v48);
  return a1;
}
