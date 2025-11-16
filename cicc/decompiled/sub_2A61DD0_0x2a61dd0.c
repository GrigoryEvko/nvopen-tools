// Function: sub_2A61DD0
// Address: 0x2a61dd0
//
char __fastcall sub_2A61DD0(__int64 a1, int a2, __int64 a3)
{
  __int64 v5; // rax
  unsigned int v6; // r9d
  unsigned __int64 v7; // r8
  bool v8; // cc
  bool v9; // al
  bool v10; // dl
  __int64 v11; // rdi
  __int64 *v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  unsigned int v15; // eax
  unsigned int v16; // eax
  _QWORD *v17; // rax
  unsigned __int64 v18; // rax
  unsigned __int64 v20; // [rsp+8h] [rbp-98h]
  unsigned int v21; // [rsp+14h] [rbp-8Ch]
  bool v22; // [rsp+14h] [rbp-8Ch]
  const void **v23; // [rsp+18h] [rbp-88h]
  __int64 v24; // [rsp+28h] [rbp-78h] BYREF
  unsigned __int64 v25; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v26; // [rsp+38h] [rbp-68h]
  unsigned __int64 v27; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v28; // [rsp+48h] [rbp-58h]
  unsigned __int64 v29; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v30; // [rsp+58h] [rbp-48h]
  unsigned __int64 v31; // [rsp+60h] [rbp-40h] BYREF
  unsigned int v32; // [rsp+68h] [rbp-38h]

  LOBYTE(v5) = *(_BYTE *)a3;
  if ( (unsigned __int8)(*(_BYTE *)a3 - 4) > 1u )
    goto LABEL_39;
  v23 = (const void **)(a3 + 8);
  v30 = *(_DWORD *)(a3 + 16);
  if ( v30 > 0x40 )
    sub_C43780((__int64)&v29, v23);
  else
    v29 = *(_QWORD *)(a3 + 8);
  sub_C46A40((__int64)&v29, 1);
  v6 = v30;
  v7 = v29;
  v30 = 0;
  v8 = *(_DWORD *)(a3 + 32) <= 0x40u;
  v26 = v6;
  v25 = v29;
  if ( v8 )
  {
    v10 = *(_QWORD *)(a3 + 24) == v29;
  }
  else
  {
    v20 = v29;
    v21 = v6;
    v9 = sub_C43C50(a3 + 24, (const void **)&v25);
    v7 = v20;
    v6 = v21;
    v10 = v9;
  }
  if ( v6 > 0x40 )
  {
    if ( v7 )
    {
      v22 = v10;
      j_j___libc_free_0_0(v7);
      v10 = v22;
      if ( v30 > 0x40 )
      {
        if ( v29 )
        {
          j_j___libc_free_0_0(v29);
          v10 = v22;
        }
      }
    }
  }
  LOBYTE(v5) = *(_BYTE *)a3;
  if ( v10 )
  {
LABEL_39:
    if ( (_BYTE)v5 == 3 )
    {
      v11 = *(_QWORD *)(a3 + 8);
      v5 = *(_QWORD *)(v11 + 8);
      if ( *(_BYTE *)(v5 + 8) == 14 )
      {
        LOBYTE(v5) = sub_AC30F0(v11);
        if ( (_BYTE)v5 )
        {
          LOBYTE(v5) = sub_B2D7C0(a1, a2, 43);
          if ( !(_BYTE)v5 )
          {
            v12 = (__int64 *)sub_B2BE50(a1);
            v13 = sub_A778C0(v12, 43, 0);
            LOBYTE(v5) = sub_B2CCF0(a1, a2, v13);
          }
        }
      }
    }
  }
  else if ( (_BYTE)v5 != 5 )
  {
    v24 = sub_B2D7B0(a1, a2, 97);
    v26 = *(_DWORD *)(a3 + 16);
    if ( v26 > 0x40 )
      sub_C43780((__int64)&v25, v23);
    else
      v25 = *(_QWORD *)(a3 + 8);
    v28 = *(_DWORD *)(a3 + 32);
    if ( v28 > 0x40 )
      sub_C43780((__int64)&v27, (const void **)(a3 + 24));
    else
      v27 = *(_QWORD *)(a3 + 24);
    if ( v24 )
    {
      v14 = sub_A72AA0(&v24);
      sub_AB2160((__int64)&v29, (__int64)&v25, v14, 0);
      if ( v26 > 0x40 && v25 )
        j_j___libc_free_0_0(v25);
      v25 = v29;
      v15 = v30;
      v30 = 0;
      v26 = v15;
      if ( v28 > 0x40 && v27 )
        j_j___libc_free_0_0(v27);
      v27 = v31;
      v16 = v32;
      v32 = 0;
      v28 = v16;
      sub_969240((__int64 *)&v31);
      sub_969240((__int64 *)&v29);
    }
    v17 = (_QWORD *)sub_B2BE50(a1);
    v18 = sub_A789D0(v17, 97, (__int64)&v25);
    sub_B2CCF0(a1, a2, v18);
    sub_969240((__int64 *)&v27);
    LOBYTE(v5) = sub_969240((__int64 *)&v25);
  }
  return v5;
}
