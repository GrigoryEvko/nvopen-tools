// Function: sub_32E2DA0
// Address: 0x32e2da0
//
__int64 __fastcall sub_32E2DA0(__int64 a1, __int64 a2, int a3, int a4, unsigned __int8 a5)
{
  char v7; // si
  char v8; // al
  __int64 v9; // rdi
  unsigned int v10; // r13d
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rax
  __int64 v15; // [rsp+10h] [rbp-78h] BYREF
  unsigned __int64 v16; // [rsp+18h] [rbp-70h] BYREF
  unsigned int v17; // [rsp+20h] [rbp-68h]
  unsigned __int64 v18; // [rsp+28h] [rbp-60h] BYREF
  unsigned int v19; // [rsp+30h] [rbp-58h]
  __int64 v20; // [rsp+38h] [rbp-50h] BYREF
  char v21; // [rsp+40h] [rbp-48h]
  char v22; // [rsp+41h] [rbp-47h]
  __int64 v23; // [rsp+48h] [rbp-40h]
  int v24; // [rsp+50h] [rbp-38h]
  __int64 v25; // [rsp+58h] [rbp-30h]
  int v26; // [rsp+60h] [rbp-28h]

  v7 = *(_BYTE *)(a1 + 34);
  v8 = *(_BYTE *)(a1 + 33);
  v23 = 0;
  v20 = *(_QWORD *)a1;
  v9 = *(_QWORD *)(a1 + 8);
  v21 = v7;
  v22 = v8;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v17 = 1;
  v16 = 0;
  v19 = 1;
  v18 = 0;
  v10 = sub_3475BB0(v9, a2, a3, a4, (unsigned int)&v16, (unsigned int)&v18, (__int64)&v20, 0, a5);
  if ( (_BYTE)v10 )
  {
    if ( *(_DWORD *)(a2 + 24) != 328 )
    {
      v15 = a2;
      sub_32B3B20(a1 + 568, &v15);
      if ( *(int *)(a2 + 88) < 0 )
      {
        *(_DWORD *)(a2 + 88) = *(_DWORD *)(a1 + 48);
        v14 = *(unsigned int *)(a1 + 48);
        if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 52) )
        {
          sub_C8D5F0(a1 + 40, (const void *)(a1 + 56), v14 + 1, 8u, v12, v13);
          v14 = *(unsigned int *)(a1 + 48);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v14) = a2;
        ++*(_DWORD *)(a1 + 48);
      }
    }
    sub_32D0190(a1, &v20);
  }
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  if ( v17 > 0x40 && v16 )
    j_j___libc_free_0_0(v16);
  return v10;
}
