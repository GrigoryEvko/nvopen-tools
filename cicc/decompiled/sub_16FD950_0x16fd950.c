// Function: sub_16FD950
// Address: 0x16fd950
//
__int64 __fastcall sub_16FD950(__int64 **a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 *v13; // rdi
  __int64 v14; // rax
  __int64 v16; // [rsp-58h] [rbp-58h] BYREF
  __int64 *v17; // [rsp-40h] [rbp-40h]
  __int64 v18; // [rsp-30h] [rbp-30h] BYREF

  if ( *(_BYTE *)(**a1 + 74) )
    return 0;
  while ( 1 )
  {
    v13 = a1[14];
    if ( !v13 )
    {
      v14 = sub_16FC3B0((__int64)a1, a2, a3, a4, a5);
      a1[14] = (__int64 *)v14;
      v13 = (__int64 *)v14;
    }
    (*(void (__fastcall **)(__int64 *))(*v13 + 8))(v13);
    v9 = *(_DWORD *)sub_16FC330(a1, a2, v6, v7, v8);
    if ( v9 == 2 )
      return 0;
    if ( v9 != 6 )
      break;
    a2 = (unsigned __int64)a1;
    sub_16FC210((__int64)&v16, (unsigned __int64 **)a1, v10, v11, v12);
    if ( v17 != &v18 )
    {
      a2 = v18 + 1;
      j_j___libc_free_0(v17, v18 + 1);
    }
    if ( *(_BYTE *)(**a1 + 74) )
      return 0;
  }
  return 1;
}
