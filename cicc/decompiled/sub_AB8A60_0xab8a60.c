// Function: sub_AB8A60
// Address: 0xab8a60
//
__int64 __fastcall sub_AB8A60(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 *v5; // r14
  __int64 *v6; // rax
  unsigned int v7; // edx
  __int64 v8; // rax
  __int64 *v9; // rax
  unsigned int v10; // ebx
  __int64 *v11; // rax
  unsigned int v12; // ebx
  unsigned int v13; // eax
  int v14; // eax
  int v15; // eax
  __int64 v16; // rcx
  __int64 v17; // r8
  unsigned int v18; // edx
  unsigned __int64 v19; // rax
  __int64 v20; // rdx
  unsigned int v21; // edx
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  unsigned int v25; // eax
  char v26; // [rsp+7h] [rbp-129h]
  char v27; // [rsp+7h] [rbp-129h]
  __int64 v28[2]; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v29[2]; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v30[2]; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v31[2]; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v32; // [rsp+80h] [rbp-B0h] BYREF
  unsigned int v33; // [rsp+88h] [rbp-A8h]
  __int64 v34; // [rsp+90h] [rbp-A0h] BYREF
  int v35; // [rsp+98h] [rbp-98h]
  __int64 v36; // [rsp+A0h] [rbp-90h] BYREF
  int v37; // [rsp+A8h] [rbp-88h]
  __int64 v38; // [rsp+B0h] [rbp-80h] BYREF
  int v39; // [rsp+B8h] [rbp-78h]
  __int64 v40; // [rsp+C0h] [rbp-70h] BYREF
  unsigned int v41; // [rsp+C8h] [rbp-68h]
  __int64 v42; // [rsp+D0h] [rbp-60h] BYREF
  __int64 v43; // [rsp+E0h] [rbp-50h] BYREF
  unsigned int v44; // [rsp+E8h] [rbp-48h]
  __int64 v45; // [rsp+F0h] [rbp-40h] BYREF
  int v46; // [rsp+F8h] [rbp-38h]

  if ( sub_AAF7D0(a2) || sub_AAF7D0((__int64)a3) )
  {
    sub_AADB10(a1, *(_DWORD *)(a2 + 8), 0);
    return a1;
  }
  if ( !sub_9876C0((__int64 *)a2) || !sub_9876C0(a3) )
  {
    if ( sub_9876C0(a3) )
    {
      v9 = sub_9876C0(a3);
      v10 = *((_DWORD *)v9 + 2);
      if ( !v10 )
      {
LABEL_14:
        sub_AB8340(a1, a2);
        return a1;
      }
      if ( v10 <= 0x40 )
      {
        if ( *v9 == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v10) )
          goto LABEL_14;
      }
      else if ( v10 == (unsigned int)sub_C445E0(v9) )
      {
        goto LABEL_14;
      }
    }
    if ( sub_9876C0((__int64 *)a2) )
    {
      v11 = sub_9876C0((__int64 *)a2);
      v12 = *((_DWORD *)v11 + 2);
      if ( !v12 )
      {
LABEL_20:
        sub_AB8340(a1, (__int64)a3);
        return a1;
      }
      if ( v12 <= 0x40 )
      {
        if ( *v11 == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v12) )
          goto LABEL_20;
      }
      else if ( v12 == (unsigned int)sub_C445E0(v11) )
      {
        goto LABEL_20;
      }
    }
    sub_AB0A90((__int64)v28, a2);
    sub_AB0A90((__int64)v30, (__int64)a3);
    sub_9865C0((__int64)&v43, (__int64)v28);
    sub_9865C0((__int64)&v45, (__int64)v29);
    sub_C7BDB0(&v43, v30);
    v13 = v44;
    v44 = 0;
    v33 = v13;
    v32 = v43;
    v35 = v46;
    v34 = v45;
    sub_969240(&v43);
    sub_AAF050((__int64)&v36, (__int64)&v32, 0);
    if ( *(_DWORD *)(a2 + 8) == 1 )
      goto LABEL_23;
    sub_9865C0((__int64)&v40, (__int64)v28);
    v18 = v41;
    if ( v41 > 0x40 )
    {
      sub_C43D10(&v40, v28, v41, v16, v17);
      v21 = v41;
      v19 = v40;
      v41 = 0;
      v44 = v21;
      v43 = v40;
      if ( v21 > 0x40 )
      {
        v26 = sub_C446F0(&v43, v31);
        goto LABEL_29;
      }
    }
    else
    {
      v44 = v41;
      v41 = 0;
      v19 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v18) & ~v40;
      if ( !v18 )
        v19 = 0;
      v40 = v19;
      v43 = v19;
    }
    v26 = (v19 & ~v31[0]) == 0;
LABEL_29:
    sub_969240(&v43);
    sub_969240(&v40);
    if ( v26 )
    {
      v20 = a2;
      a2 = (__int64)a3;
    }
    else
    {
      sub_9865C0((__int64)&v40, (__int64)v30);
      sub_987160((__int64)&v40, (__int64)v30, v22, v23, v24);
      v25 = v41;
      v41 = 0;
      v44 = v25;
      v43 = v40;
      if ( v25 <= 0x40 )
        v27 = (v40 & ~v29[0]) == 0;
      else
        v27 = sub_C446F0(&v43, v29);
      sub_969240(&v43);
      sub_969240(&v40);
      if ( !v27 )
        goto LABEL_23;
      v20 = (__int64)a3;
    }
    sub_AB51C0((__int64)&v40, a2, v20);
    sub_AB2160((__int64)&v43, (__int64)&v36, (__int64)&v40, 1u);
    sub_AAD5C0(&v36, &v43);
    sub_969240(&v45);
    sub_969240(&v43);
    sub_969240(&v42);
    sub_969240(&v40);
LABEL_23:
    v14 = v37;
    v37 = 0;
    *(_DWORD *)(a1 + 8) = v14;
    *(_QWORD *)a1 = v36;
    v15 = v39;
    v39 = 0;
    *(_DWORD *)(a1 + 24) = v15;
    *(_QWORD *)(a1 + 16) = v38;
    sub_969240(&v38);
    sub_969240(&v36);
    sub_969240(&v34);
    sub_969240(&v32);
    sub_969240(v31);
    sub_969240(v30);
    sub_969240(v29);
    sub_969240(v28);
    return a1;
  }
  v5 = sub_9876C0(a3);
  v6 = sub_9876C0((__int64 *)a2);
  sub_9865C0((__int64)&v40, (__int64)v6);
  v7 = v41;
  if ( v41 > 0x40 )
  {
    sub_C43C10(&v40, v5);
    v7 = v41;
    v8 = v40;
  }
  else
  {
    v8 = *v5 ^ v40;
    v40 = v8;
  }
  v44 = v7;
  v43 = v8;
  v41 = 0;
  sub_AADBC0(a1, &v43);
  sub_969240(&v43);
  sub_969240(&v40);
  return a1;
}
