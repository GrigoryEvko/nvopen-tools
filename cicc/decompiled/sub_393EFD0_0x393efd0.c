// Function: sub_393EFD0
// Address: 0x393efd0
//
char *__fastcall sub_393EFD0(const char *a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char *v6; // r12
  __int64 v7; // r13
  size_t v8; // rdx
  __int64 v9; // rax
  unsigned __int64 v10; // rax
  __int64 v12; // r13
  __int64 v13; // rdi
  __int64 v14; // r14
  __int64 v15; // rdx
  char *(*v16)(); // rcx
  char *v17; // rax
  unsigned __int64 *v18; // [rsp+0h] [rbp-90h] BYREF
  __int16 v19; // [rsp+10h] [rbp-80h]
  unsigned __int64 v20[2]; // [rsp+20h] [rbp-70h] BYREF
  __int64 v21; // [rsp+30h] [rbp-60h] BYREF
  _QWORD v22[4]; // [rsp+40h] [rbp-50h] BYREF
  int v23; // [rsp+60h] [rbp-30h]
  unsigned __int64 **v24; // [rsp+68h] [rbp-28h]

  v6 = (char *)a1;
  v7 = a2[9];
  if ( v7 )
  {
    a1 = (const char *)a2[9];
    v8 = strlen(a1);
    v9 = v8 + 1;
  }
  else
  {
    v9 = 1;
    v8 = 0;
  }
  v10 = v7 + v9;
  if ( v10 > a2[10] )
  {
    v12 = sub_393D180((__int64)a1, (__int64)a2, v8, a4, a5, a6);
    (*(void (__fastcall **)(unsigned __int64 *, __int64, __int64))(*(_QWORD *)v12 + 32LL))(v20, v12, 4);
    v13 = a2[6];
    v18 = v20;
    v19 = 260;
    v14 = a2[5];
    v15 = 14;
    v16 = *(char *(**)())(*(_QWORD *)v13 + 16LL);
    v17 = "Unknown buffer";
    if ( v16 != sub_12BCB10 )
      v17 = (char *)((__int64 (__fastcall *)(__int64, __int64, __int64))v16)(v13, v12, 14);
    v22[2] = v17;
    v22[1] = 7;
    v24 = &v18;
    v22[0] = &unk_49ECF18;
    v22[3] = v15;
    v23 = 0;
    sub_16027F0(v14, (__int64)v22);
    if ( (__int64 *)v20[0] != &v21 )
      j_j___libc_free_0(v20[0]);
    *((_QWORD *)v6 + 1) = v12;
    v6[16] |= 1u;
    *(_DWORD *)v6 = 4;
    return v6;
  }
  else
  {
    a2[9] = v10;
    *(_QWORD *)v6 = v7;
    v6[16] &= ~1u;
    *((_QWORD *)v6 + 1) = v8;
    return v6;
  }
}
