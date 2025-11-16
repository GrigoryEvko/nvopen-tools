// Function: sub_EAFAF0
// Address: 0xeafaf0
//
__int64 __fastcall sub_EAFAF0(__int64 a1, const char **a2, const char **a3)
{
  __int64 v4; // rax
  __int64 v5; // rax
  unsigned int v6; // r15d
  const char *v7; // rax
  bool v8; // zf
  __int64 result; // rax
  unsigned int v10; // eax
  const char *v11; // rdi
  const char *v12; // rdi
  const char **v13; // rax
  __int64 v14; // [rsp+8h] [rbp-78h]
  unsigned __int8 v15; // [rsp+8h] [rbp-78h]
  char *v16; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v17; // [rsp+18h] [rbp-68h]
  const char *v18; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v19; // [rsp+28h] [rbp-58h]
  char v20; // [rsp+40h] [rbp-40h]
  char v21; // [rsp+41h] [rbp-3Fh]

  if ( *(_DWORD *)sub_ECD7B0(a1) != 4 && *(_DWORD *)sub_ECD7B0(a1) != 5 )
  {
    v21 = 1;
    v18 = "unknown token in expression";
    v20 = 3;
    return sub_ECE0E0(a1, &v18, 0, 0);
  }
  v4 = sub_ECD7B0(a1);
  v14 = sub_ECD6A0(v4);
  v5 = sub_ECD7B0(a1);
  v17 = *(_DWORD *)(v5 + 32);
  if ( v17 > 0x40 )
    sub_C43780((__int64)&v16, (const void **)(v5 + 24));
  else
    v16 = *(char **)(v5 + 24);
  sub_EABFE0(a1);
  v6 = v17;
  if ( v17 <= 0x40 )
  {
    v7 = 0;
    v8 = v16 == 0;
    *a2 = 0;
    if ( !v8 )
      v7 = v16;
LABEL_8:
    *a3 = v7;
    result = 0;
    goto LABEL_9;
  }
  v10 = v6 - sub_C444A0((__int64)&v16);
  if ( v10 <= 0x80 )
  {
    if ( v10 > 0x40 )
    {
      sub_C48300((__int64)&v18, (__int64)&v16, v6 - 64);
      if ( v19 <= 0x40 )
      {
        *a2 = v18;
      }
      else
      {
        v11 = v18;
        *a2 = *(const char **)v18;
        j_j___libc_free_0_0(v11);
      }
      sub_C443A0((__int64)&v18, (__int64)&v16, 0x40u);
      if ( v19 <= 0x40 )
      {
        *a3 = v18;
      }
      else
      {
        v12 = v18;
        *a3 = *(const char **)v18;
        j_j___libc_free_0_0(v12);
      }
      v6 = v17;
      result = 0;
      goto LABEL_9;
    }
    v13 = (const char **)v16;
    *a2 = 0;
    v7 = *v13;
    goto LABEL_8;
  }
  v21 = 1;
  v18 = "out of range literal value";
  v20 = 3;
  result = sub_ECDA70(a1, v14, &v18, 0, 0);
  v6 = v17;
LABEL_9:
  if ( v6 > 0x40 )
  {
    if ( v16 )
    {
      v15 = result;
      j_j___libc_free_0_0(v16);
      return v15;
    }
  }
  return result;
}
