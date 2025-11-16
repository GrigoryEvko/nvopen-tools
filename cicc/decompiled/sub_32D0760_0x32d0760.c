// Function: sub_32D0760
// Address: 0x32d0760
//
__int64 __fastcall sub_32D0760(__int64 a1, __int64 a2, int a3, int a4, int a5, unsigned __int8 a6)
{
  char v8; // si
  char v9; // al
  __int64 v10; // rdi
  unsigned int v11; // r13d
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rax
  __int64 v16; // [rsp+10h] [rbp-78h] BYREF
  unsigned __int64 v17; // [rsp+18h] [rbp-70h] BYREF
  unsigned int v18; // [rsp+20h] [rbp-68h]
  unsigned __int64 v19; // [rsp+28h] [rbp-60h]
  unsigned int v20; // [rsp+30h] [rbp-58h]
  __int64 v21; // [rsp+38h] [rbp-50h] BYREF
  char v22; // [rsp+40h] [rbp-48h]
  char v23; // [rsp+41h] [rbp-47h]
  __int64 v24; // [rsp+48h] [rbp-40h]
  int v25; // [rsp+50h] [rbp-38h]
  __int64 v26; // [rsp+58h] [rbp-30h]
  int v27; // [rsp+60h] [rbp-28h]

  v8 = *(_BYTE *)(a1 + 34);
  v9 = *(_BYTE *)(a1 + 33);
  v24 = 0;
  v21 = *(_QWORD *)a1;
  v10 = *(_QWORD *)(a1 + 8);
  v22 = v8;
  v23 = v9;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v18 = 1;
  v17 = 0;
  v20 = 1;
  v19 = 0;
  v11 = sub_347A8D0(v10, a2, a3, a4, a5, (unsigned int)&v17, (__int64)&v21, 0, a6);
  if ( (_BYTE)v11 )
  {
    if ( *(_DWORD *)(a2 + 24) != 328 )
    {
      v16 = a2;
      sub_32B3B20(a1 + 568, &v16);
      if ( *(int *)(a2 + 88) < 0 )
      {
        *(_DWORD *)(a2 + 88) = *(_DWORD *)(a1 + 48);
        v15 = *(unsigned int *)(a1 + 48);
        if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 52) )
        {
          sub_C8D5F0(a1 + 40, (const void *)(a1 + 56), v15 + 1, 8u, v13, v14);
          v15 = *(unsigned int *)(a1 + 48);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v15) = a2;
        ++*(_DWORD *)(a1 + 48);
      }
    }
    sub_32D0190(a1, &v21);
  }
  if ( v20 > 0x40 && v19 )
    j_j___libc_free_0_0(v19);
  if ( v18 > 0x40 && v17 )
    j_j___libc_free_0_0(v17);
  return v11;
}
