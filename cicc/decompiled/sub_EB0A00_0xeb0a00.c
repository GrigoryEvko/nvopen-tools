// Function: sub_EB0A00
// Address: 0xeb0a00
//
__int64 __fastcall sub_EB0A00(_QWORD *a1, __int64 a2)
{
  __int64 v4; // rdi
  unsigned int v5; // r11d
  _DWORD *v7; // rax
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 v10; // r14
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 (__fastcall *v14)(__int64); // r10
  __int64 *v15; // rax
  unsigned int v16; // eax
  unsigned __int8 v17; // [rsp+8h] [rbp-B8h]
  __int64 v18; // [rsp+8h] [rbp-B8h]
  __int64 v19; // [rsp+10h] [rbp-B0h] BYREF
  int *v20; // [rsp+18h] [rbp-A8h] BYREF
  __int64 v21; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v22; // [rsp+28h] [rbp-98h] BYREF
  _QWORD v23[3]; // [rsp+30h] [rbp-90h] BYREF
  int v24; // [rsp+48h] [rbp-78h]
  __int16 v25; // [rsp+50h] [rbp-70h]
  const char *v26; // [rsp+60h] [rbp-60h] BYREF
  __int64 *v27; // [rsp+68h] [rbp-58h] BYREF
  __int64 v28; // [rsp+78h] [rbp-48h] BYREF
  char v29; // [rsp+80h] [rbp-40h]
  char v30; // [rsp+81h] [rbp-3Fh]
  char v31; // [rsp+88h] [rbp-38h]

  v4 = a1[6];
  v20 = 0;
  v21 = sub_ECD6A0(v4);
  v26 = 0;
  if ( sub_EAC4D0((__int64)a1, &v19, (__int64)&v26) )
    return 1;
  v30 = 1;
  v29 = 3;
  v26 = "expected comma";
  if ( (unsigned __int8)sub_ECE210(a1, 26, &v26) )
    return 1;
  v30 = 1;
  v26 = "expected relocation name";
  v29 = 3;
  v7 = (_DWORD *)sub_ECD7B0(a1);
  if ( (unsigned __int8)sub_ECE0A0(a1, *v7 != 2, &v26) )
    return 1;
  v22 = sub_ECD6A0(a1[6]);
  v8 = a1[6];
  if ( *(_DWORD *)v8 == 2 )
  {
    v10 = *(_QWORD *)(v8 + 8);
    v9 = *(_QWORD *)(v8 + 16);
  }
  else
  {
    v9 = *(_QWORD *)(v8 + 16);
    v10 = *(_QWORD *)(v8 + 8);
    if ( v9 )
    {
      v11 = v9 - 1;
      if ( !v11 )
        v11 = 1;
      ++v10;
      v9 = v11 - 1;
    }
  }
  sub_EABFE0((__int64)a1);
  if ( *(_DWORD *)a1[6] == 26 )
  {
    sub_EABFE0((__int64)a1);
    v18 = sub_ECD690(a1 + 5);
    v26 = 0;
    LOBYTE(v16) = sub_EAC4D0((__int64)a1, (__int64 *)&v20, (__int64)&v26);
    v5 = v16;
    if ( (_BYTE)v16 )
      return v5;
    memset(v23, 0, sizeof(v23));
    v24 = 0;
    if ( !(unsigned __int8)sub_E81950(v20, (__int64)v23, 0, 0) )
    {
      v30 = 1;
      v26 = "expression must be relocatable";
      v29 = 3;
      return (unsigned int)sub_ECDA70(a1, v18, &v26, 0, 0);
    }
  }
  if ( (unsigned __int8)sub_ECE000(a1) )
  {
    return 1;
  }
  else
  {
    v12 = sub_ECE6C0(a1[1]);
    v13 = a1[29];
    v5 = 0;
    v14 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v13 + 1200LL);
    if ( v14 != sub_C14030 )
    {
      ((void (__fastcall *)(const char **, __int64, __int64, __int64, __int64, int *, __int64, __int64))v14)(
        &v26,
        v13,
        v19,
        v10,
        v9,
        v20,
        a2,
        v12);
      v5 = 0;
      if ( v31 )
      {
        v25 = 260;
        v23[0] = &v27;
        v15 = &v22;
        if ( !(_BYTE)v26 )
          v15 = &v21;
        v5 = sub_ECDA70(a1, *v15, v23, 0, 0);
        if ( v31 )
        {
          v31 = 0;
          if ( v27 != &v28 )
          {
            v17 = v5;
            j_j___libc_free_0(v27, v28 + 1);
            return v17;
          }
        }
      }
    }
  }
  return v5;
}
