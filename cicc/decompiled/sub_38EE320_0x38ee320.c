// Function: sub_38EE320
// Address: 0x38ee320
//
__int64 __fastcall sub_38EE320(__int64 a1, const char **a2, _QWORD *a3)
{
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rax
  __int64 v7; // rax
  unsigned int v8; // r15d
  unsigned __int64 v9; // rax
  bool v10; // zf
  __int64 result; // rax
  unsigned int v12; // eax
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  unsigned __int64 *v15; // rax
  __int64 v16; // [rsp+8h] [rbp-68h]
  unsigned __int8 v17; // [rsp+8h] [rbp-68h]
  unsigned __int64 *v18; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v19; // [rsp+18h] [rbp-58h]
  const char *v20; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v21; // [rsp+28h] [rbp-48h]
  char v22; // [rsp+30h] [rbp-40h]
  char v23; // [rsp+31h] [rbp-3Fh]

  if ( *(_DWORD *)sub_3909460(a1) != 4 && *(_DWORD *)sub_3909460(a1) != 5 )
  {
    v23 = 1;
    v20 = "unknown token in expression";
    v22 = 3;
    return sub_3909CF0(a1, &v20, 0, 0, v4, v5);
  }
  v6 = sub_3909460(a1);
  v16 = sub_39092A0(v6);
  v7 = sub_3909460(a1);
  v19 = *(_DWORD *)(v7 + 32);
  if ( v19 > 0x40 )
    sub_16A4FD0((__int64)&v18, (const void **)(v7 + 24));
  else
    v18 = *(unsigned __int64 **)(v7 + 24);
  sub_38EB180(a1);
  v8 = v19;
  if ( v19 <= 0x40 )
  {
    v9 = 0;
    v10 = v18 == 0;
    *a2 = 0;
    if ( !v10 )
      v9 = (unsigned __int64)v18;
LABEL_8:
    *a3 = v9;
    result = 0;
    goto LABEL_9;
  }
  v12 = v8 - sub_16A57B0((__int64)&v18);
  if ( v12 <= 0x80 )
  {
    if ( v12 > 0x40 )
    {
      sub_16A8130((__int64)&v20, (__int64)&v18, v8 - 64);
      if ( v21 <= 0x40 )
      {
        *a2 = v20;
      }
      else
      {
        v13 = (unsigned __int64)v20;
        *a2 = *(const char **)v20;
        j_j___libc_free_0_0(v13);
      }
      sub_16A88B0((__int64)&v20, (__int64)&v18, 0x40u);
      if ( v21 <= 0x40 )
      {
        *a3 = v20;
      }
      else
      {
        v14 = (unsigned __int64)v20;
        *a3 = *(_QWORD *)v20;
        j_j___libc_free_0_0(v14);
      }
      v8 = v19;
      result = 0;
      goto LABEL_9;
    }
    v15 = v18;
    *a2 = 0;
    v9 = *v15;
    goto LABEL_8;
  }
  v23 = 1;
  v20 = "out of range literal value";
  v22 = 3;
  result = sub_3909790(a1, v16, &v20, 0, 0);
  v8 = v19;
LABEL_9:
  if ( v8 > 0x40 )
  {
    if ( v18 )
    {
      v17 = result;
      j_j___libc_free_0_0((unsigned __int64)v18);
      return v17;
    }
  }
  return result;
}
