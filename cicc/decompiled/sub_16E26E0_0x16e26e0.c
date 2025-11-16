// Function: sub_16E26E0
// Address: 0x16e26e0
//
void __fastcall sub_16E26E0(__int64 *a1, const void *a2, unsigned __int64 a3, __int64 a4, int a5, int a6)
{
  size_t v7; // rbx
  __int64 v8; // rax
  unsigned __int64 v9; // rdx
  int v10; // r8d
  int v11; // r9d
  __int64 v12; // rdi
  const void *v13; // r15
  size_t v14; // rbx
  unsigned int v15; // ebx
  __int64 v16; // rdx
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  int v19; // r8d
  int v20; // r9d
  __int64 v21; // rdi
  const void *v22; // r15
  size_t v23; // rbx
  _BYTE *v24; // rdi
  int v25; // r8d
  int v26; // r9d
  unsigned int v27; // ebx
  _BYTE **v28; // [rsp+0h] [rbp-A0h] BYREF
  __int16 v29; // [rsp+10h] [rbp-90h]
  _BYTE *v30; // [rsp+20h] [rbp-80h] BYREF
  __int64 v31; // [rsp+28h] [rbp-78h]
  _BYTE dest[112]; // [rsp+30h] [rbp-70h] BYREF

  v7 = a3;
  v30 = dest;
  v31 = 0x4000000000LL;
  if ( a3 > 0x40 )
  {
    sub_16CD150((__int64)&v30, dest, a3, 1, a5, a6);
    v24 = &v30[(unsigned int)v31];
  }
  else
  {
    if ( !a3 )
    {
      LODWORD(v31) = 0;
      goto LABEL_4;
    }
    v24 = dest;
  }
  memcpy(v24, a2, v7);
  v27 = v31 + v7;
  LODWORD(v31) = v27;
  a3 = v27;
  if ( HIDWORD(v31) == v27 )
  {
    sub_16CD150((__int64)&v30, dest, v27 + 1LL, 1, v25, v26);
    a3 = (unsigned int)v31;
  }
LABEL_4:
  v30[a3] = 45;
  LODWORD(v31) = v31 + 1;
  v8 = sub_16E1F20(a1);
  v12 = (unsigned int)v31;
  v13 = (const void *)v8;
  v14 = v9;
  if ( v9 > HIDWORD(v31) - (unsigned __int64)(unsigned int)v31 )
  {
    sub_16CD150((__int64)&v30, dest, (unsigned int)v31 + v9, 1, v10, v11);
    v12 = (unsigned int)v31;
  }
  if ( v14 )
  {
    memcpy(&v30[v12], v13, v14);
    LODWORD(v12) = v31;
  }
  v15 = v12 + v14;
  LODWORD(v31) = v15;
  v16 = v15;
  if ( HIDWORD(v31) == v15 )
  {
    sub_16CD150((__int64)&v30, dest, v15 + 1LL, 1, v10, v11);
    v16 = (unsigned int)v31;
  }
  v30[v16] = 45;
  LODWORD(v31) = v31 + 1;
  v17 = sub_16E2230(a1);
  v21 = (unsigned int)v31;
  v22 = (const void *)v17;
  v23 = v18;
  if ( v18 > HIDWORD(v31) - (unsigned __int64)(unsigned int)v31 )
  {
    sub_16CD150((__int64)&v30, dest, (unsigned int)v31 + v18, 1, v19, v20);
    v21 = (unsigned int)v31;
  }
  if ( v23 )
  {
    memcpy(&v30[v21], v22, v23);
    LODWORD(v21) = v31;
  }
  v28 = &v30;
  LODWORD(v31) = v21 + v23;
  v29 = 262;
  sub_16E25E0(a1, (__int64)&v28);
  if ( v30 != dest )
    _libc_free((unsigned __int64)v30);
}
