// Function: sub_AAF830
// Address: 0xaaf830
//
__int64 __fastcall sub_AAF830(__int64 a1, int *a2, __int64 a3, __int64 *a4)
{
  __int64 result; // rax
  __int64 *v8; // rax
  __int64 v9; // rsi
  unsigned int v10; // eax
  bool v11; // cc
  __int64 v12; // rdx
  __int64 v13; // rcx
  unsigned int v14; // r8d
  __int64 v15; // rdx
  __int64 v16; // rcx
  unsigned int v17; // r8d
  int v18; // eax
  __int64 v19; // rsi
  bool v20; // r10
  unsigned int v21; // eax
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  char v25; // [rsp+Fh] [rbp-61h]
  __int64 *v26; // [rsp+18h] [rbp-58h]
  __int64 v27; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v28; // [rsp+28h] [rbp-48h]
  __int64 v29; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v30; // [rsp+38h] [rbp-38h]

  v30 = *(_DWORD *)(a1 + 8);
  if ( v30 > 0x40 )
    sub_C43690(&v29, 0, 0);
  else
    v29 = 0;
  if ( *((_DWORD *)a4 + 2) > 0x40u && *a4 )
    j_j___libc_free_0_0(*a4);
  *a4 = v29;
  *((_DWORD *)a4 + 2) = v30;
  if ( sub_AAF760(a1) || sub_AAF7D0(a1) )
  {
    *a2 = 35 - ((sub_AAF7D0(a1) == 0) - 1);
    v30 = *(_DWORD *)(a1 + 8);
    if ( v30 > 0x40 )
      sub_C43690(&v29, 0, 0);
    else
      v29 = 0;
    if ( *(_DWORD *)(a3 + 8) > 0x40u )
    {
      if ( *(_QWORD *)a3 )
        j_j___libc_free_0_0(*(_QWORD *)a3);
    }
    *(_QWORD *)a3 = v29;
    result = v30;
    *(_DWORD *)(a3 + 8) = v30;
    return result;
  }
  v8 = sub_9876C0((__int64 *)a1);
  if ( !v8 )
  {
    v26 = (__int64 *)(a1 + 16);
    sub_9865C0((__int64)&v27, a1 + 16);
    v9 = 1;
    sub_C46A40(&v27, 1);
    v10 = v28;
    v11 = *(_DWORD *)(a1 + 8) <= 0x40u;
    v28 = 0;
    v30 = v10;
    v29 = v27;
    if ( v11 )
    {
      v25 = *(_QWORD *)a1 == v27;
    }
    else
    {
      v9 = (__int64)&v29;
      v25 = sub_C43C50(a1, &v29);
    }
    sub_969240(&v29);
    sub_969240(&v27);
    if ( v25 )
    {
      *a2 = 33;
    }
    else if ( (unsigned __int8)sub_986B30((__int64 *)a1, v9, v12, v13, v14) )
    {
      *a2 = 40;
    }
    else
    {
      if ( !sub_9867B0(a1) )
      {
        if ( (unsigned __int8)sub_986B30(v26, v9, v15, v16, v17) )
        {
          v18 = 39;
        }
        else
        {
          v20 = sub_9867B0((__int64)v26);
          v18 = 35;
          if ( !v20 )
          {
            *a2 = 36;
            sub_9865C0((__int64)&v27, (__int64)v26);
            sub_C46B40(&v27, a1);
            v21 = v28;
            v28 = 0;
            v30 = v21;
            v29 = v27;
            sub_AAD550((__int64 *)a3, &v29);
            sub_969240(&v29);
            sub_969240(&v27);
            sub_9865C0((__int64)&v27, a1);
            sub_AADAA0((__int64)&v29, (__int64)&v27, v22, v23, v24);
            sub_AAD550(a4, &v29);
            sub_969240(&v29);
            return sub_969240(&v27);
          }
        }
        *a2 = v18;
        v19 = a1;
        return sub_AAD590(a3, v19);
      }
      *a2 = 36;
    }
    v19 = a1 + 16;
    return sub_AAD590(a3, v19);
  }
  *a2 = 32;
  if ( *(_DWORD *)(a3 + 8) > 0x40u || *((_DWORD *)v8 + 2) > 0x40u )
    return sub_C43990(a3, v8);
  *(_QWORD *)a3 = *v8;
  result = *((unsigned int *)v8 + 2);
  *(_DWORD *)(a3 + 8) = result;
  return result;
}
