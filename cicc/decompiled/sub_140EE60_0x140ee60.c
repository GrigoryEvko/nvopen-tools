// Function: sub_140EE60
// Address: 0x140ee60
//
__int64 __fastcall sub_140EE60(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // r8d
  unsigned int v5; // edx
  unsigned int v6; // r13d
  char v7; // al
  unsigned int v8; // r14d
  char v9; // al
  unsigned int v10; // eax
  unsigned int v11; // eax
  __int64 v13; // rcx
  __int64 v14; // rax
  unsigned int v15; // eax
  unsigned int v16; // eax
  unsigned int v17; // [rsp+4h] [rbp-9Ch]
  unsigned int v18; // [rsp+4h] [rbp-9Ch]
  unsigned int v19; // [rsp+8h] [rbp-98h]
  unsigned int v20; // [rsp+8h] [rbp-98h]
  __int64 v21; // [rsp+10h] [rbp-90h] BYREF
  unsigned int v22; // [rsp+18h] [rbp-88h]
  __int64 v23; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v24; // [rsp+28h] [rbp-78h]
  __int64 v25; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v26; // [rsp+38h] [rbp-68h]
  __int64 v27; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v28; // [rsp+48h] [rbp-58h]
  __int64 v29; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v30; // [rsp+58h] [rbp-48h]
  __int64 v31; // [rsp+60h] [rbp-40h] BYREF
  unsigned int v32; // [rsp+68h] [rbp-38h]

  sub_140E6D0((__int64)&v25, a2, *(_QWORD **)(a3 - 48));
  sub_140E6D0((__int64)&v29, a2, *(_QWORD **)(a3 - 24));
  v4 = v26;
  if ( v26 <= 1 )
  {
LABEL_26:
    v6 = v32;
LABEL_27:
    *(_DWORD *)(a1 + 8) = 1;
    *(_QWORD *)a1 = 0;
    *(_DWORD *)(a1 + 24) = 1;
    *(_QWORD *)(a1 + 16) = 0;
    goto LABEL_28;
  }
  v5 = v28;
  v6 = v32;
  if ( v28 <= 1 || v30 <= 1 || v32 <= 1 )
    goto LABEL_27;
  if ( v26 <= 0x40 )
  {
    if ( v25 != v29 )
      goto LABEL_7;
  }
  else
  {
    v17 = v28;
    v19 = v26;
    v7 = sub_16A5220(&v25, &v29);
    v4 = v19;
    v5 = v17;
    if ( !v7 )
      goto LABEL_7;
  }
  if ( v5 <= 0x40 )
  {
    v13 = v31;
    if ( v27 == v31 )
      goto LABEL_45;
  }
  else
  {
    v18 = v5;
    v20 = v4;
    if ( (unsigned __int8)sub_16A5220(&v27, &v31) )
    {
      v13 = v27;
      v5 = v18;
      v4 = v20;
LABEL_45:
      v14 = v25;
      *(_DWORD *)(a1 + 8) = v4;
      v26 = 0;
      *(_QWORD *)a1 = v14;
      *(_DWORD *)(a1 + 24) = v5;
      *(_QWORD *)(a1 + 16) = v13;
      v28 = 0;
      goto LABEL_28;
    }
  }
LABEL_7:
  sub_140AE80((__int64)&v21, &v25);
  sub_140AE80((__int64)&v23, &v29);
  v8 = v22;
  if ( v22 <= 0x40 )
  {
    if ( v21 == v23 )
      goto LABEL_12;
  }
  else if ( (unsigned __int8)sub_16A5220(&v21, &v23) )
  {
LABEL_12:
    v10 = v26;
    v26 = 0;
    *(_DWORD *)(a1 + 8) = v10;
    *(_QWORD *)a1 = v25;
    v11 = v28;
    v28 = 0;
    *(_DWORD *)(a1 + 24) = v11;
    *(_QWORD *)(a1 + 16) = v27;
    goto LABEL_13;
  }
  v9 = *(_BYTE *)(a2 + 16);
  if ( v9 == 1 )
  {
    if ( (int)sub_16AEA10(&v21, &v23) < 0 )
      goto LABEL_12;
    goto LABEL_51;
  }
  if ( v9 != 2 )
  {
    if ( v24 > 0x40 && v23 )
    {
      j_j___libc_free_0_0(v23);
      v8 = v22;
    }
    if ( v8 > 0x40 && v21 )
      j_j___libc_free_0_0(v21);
    goto LABEL_26;
  }
  if ( (int)sub_16AEA10(&v21, &v23) > 0 )
    goto LABEL_12;
LABEL_51:
  v15 = v30;
  v30 = 0;
  *(_DWORD *)(a1 + 8) = v15;
  *(_QWORD *)a1 = v29;
  v16 = v32;
  v32 = 0;
  *(_DWORD *)(a1 + 24) = v16;
  *(_QWORD *)(a1 + 16) = v31;
LABEL_13:
  if ( v24 > 0x40 && v23 )
  {
    j_j___libc_free_0_0(v23);
    v8 = v22;
  }
  if ( v8 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
  v6 = v32;
LABEL_28:
  if ( v6 > 0x40 && v31 )
    j_j___libc_free_0_0(v31);
  if ( v30 > 0x40 && v29 )
    j_j___libc_free_0_0(v29);
  if ( v28 > 0x40 && v27 )
    j_j___libc_free_0_0(v27);
  if ( v26 > 0x40 && v25 )
    j_j___libc_free_0_0(v25);
  return a1;
}
