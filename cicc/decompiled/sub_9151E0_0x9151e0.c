// Function: sub_9151E0
// Address: 0x9151e0
//
__int64 __fastcall sub_9151E0(_QWORD *a1)
{
  __int64 v1; // r12
  __int64 v2; // r13
  __int64 v3; // r15
  __int64 v4; // r14
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // rax
  _BYTE *v10; // rsi
  __int64 v11; // r14
  __int64 v12; // r13
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  _BYTE *v18; // rsi
  __int64 v19; // rax
  _BYTE *v20; // rsi
  _BYTE *v21; // rsi
  __int64 v22; // rax
  __int64 result; // rax
  __int64 v24; // [rsp+8h] [rbp-58h] BYREF
  __int64 v25; // [rsp+10h] [rbp-50h] BYREF
  _BYTE *v26; // [rsp+18h] [rbp-48h]
  _BYTE *v27; // [rsp+20h] [rbp-40h]

  v1 = sub_BA8E40(*a1, "nvvmir.version", 14);
  v2 = sub_BCB2D0(a1[43]);
  v3 = sub_AD64C0(v2, 2, 0);
  v25 = 0;
  v26 = 0;
  v4 = sub_AD64C0(v2, 0, 0);
  v27 = 0;
  v24 = sub_B98A20(v3, 0, v5, v6);
  sub_914280((__int64)&v25, 0, &v24);
  v9 = sub_B98A20(v4, 0, v7, v8);
  v10 = v26;
  v24 = v9;
  if ( v26 == v27 )
  {
    sub_914280((__int64)&v25, v26, &v24);
  }
  else
  {
    if ( v26 )
    {
      *(_QWORD *)v26 = v9;
      v10 = v26;
    }
    v26 = v10 + 8;
  }
  if ( !a1[46] )
    goto LABEL_15;
  v11 = sub_AD64C0(v2, 3, 0);
  v12 = sub_AD64C0(v2, 2, 0);
  v15 = sub_B98A20(v11, 2, v13, v14);
  v18 = v26;
  v24 = v15;
  if ( v26 == v27 )
  {
    sub_914280((__int64)&v25, v26, &v24);
  }
  else
  {
    if ( v26 )
    {
      *(_QWORD *)v26 = v15;
      v18 = v26;
    }
    v18 += 8;
    v26 = v18;
  }
  v19 = sub_B98A20(v12, v18, v16, v17);
  v20 = v26;
  v24 = v19;
  if ( v26 == v27 )
  {
    sub_914280((__int64)&v25, v26, &v24);
LABEL_15:
    v21 = v26;
    goto LABEL_16;
  }
  if ( v26 )
  {
    *(_QWORD *)v26 = v19;
    v20 = v26;
  }
  v21 = v20 + 8;
  v26 = v21;
LABEL_16:
  v22 = sub_B9C770(a1[43], v25, (__int64)&v21[-v25] >> 3, 0, 1);
  result = sub_B979A0(v1, v22);
  if ( v25 )
    return j_j___libc_free_0(v25, &v27[-v25]);
  return result;
}
