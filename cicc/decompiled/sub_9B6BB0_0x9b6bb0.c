// Function: sub_9B6BB0
// Address: 0x9b6bb0
//
__int64 __fastcall sub_9B6BB0(
        unsigned int a1,
        unsigned int a2,
        void (__fastcall *a3)(int **, __int64 *, __int64 *, char *))
{
  unsigned int v3; // eax
  int v5; // ecx
  char v6; // [rsp+7h] [rbp-39h] BYREF
  __int64 v7; // [rsp+8h] [rbp-38h]
  __int64 v8; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v9; // [rsp+18h] [rbp-28h]
  __int64 v10; // [rsp+20h] [rbp-20h] BYREF
  unsigned int v11; // [rsp+28h] [rbp-18h]
  int *v12; // [rsp+30h] [rbp-10h] BYREF
  unsigned int v13; // [rsp+38h] [rbp-8h]

  v8 = a1;
  v9 = 32;
  v11 = 32;
  v10 = a2;
  a3(&v12, &v8, &v10, &v6);
  if ( v6 )
  {
    BYTE4(v7) = 0;
    v3 = v13;
  }
  else
  {
    v3 = v13;
    if ( v13 <= 0x40 )
    {
      v5 = 0;
      if ( v13 )
        v5 = (__int64)((_QWORD)v12 << (64 - (unsigned __int8)v13)) >> (64 - (unsigned __int8)v13);
    }
    else
    {
      v5 = *v12;
    }
    LODWORD(v7) = v5;
    BYTE4(v7) = 1;
  }
  if ( v3 > 0x40 && v12 )
    j_j___libc_free_0_0(v12);
  if ( v11 > 0x40 && v10 )
    j_j___libc_free_0_0(v10);
  if ( v9 > 0x40 && v8 )
    j_j___libc_free_0_0(v8);
  return v7;
}
