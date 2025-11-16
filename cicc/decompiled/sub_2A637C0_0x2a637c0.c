// Function: sub_2A637C0
// Address: 0x2a637c0
//
__int64 __fastcall sub_2A637C0(__int64 a1, __int64 a2, __int64 a3)
{
  const void **v4; // r13
  unsigned int v5; // edx
  unsigned __int64 v6; // r8
  bool v7; // cc
  bool v8; // al
  unsigned int v9; // edx
  unsigned __int64 v10; // r14
  bool v11; // al
  bool v12; // bl
  unsigned __int64 v14; // [rsp+0h] [rbp-60h]
  unsigned int v15; // [rsp+Ch] [rbp-54h]
  bool v16; // [rsp+Ch] [rbp-54h]
  unsigned int v17; // [rsp+Ch] [rbp-54h]
  const void *v18; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v19; // [rsp+18h] [rbp-48h]
  const void *v20; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v21; // [rsp+28h] [rbp-38h]

  if ( *(_BYTE *)a2 == 2 )
    return *(_QWORD *)(a2 + 8);
  if ( (unsigned __int8)(*(_BYTE *)a2 - 4) > 1u )
    return 0;
  v4 = (const void **)(a2 + 8);
  v21 = *(_DWORD *)(a2 + 16);
  if ( v21 > 0x40 )
    sub_C43780((__int64)&v20, (const void **)(a2 + 8));
  else
    v20 = *(const void **)(a2 + 8);
  sub_C46A40((__int64)&v20, 1);
  v5 = v21;
  v6 = (unsigned __int64)v20;
  v21 = 0;
  v7 = *(_DWORD *)(a2 + 32) <= 0x40u;
  v19 = v5;
  v18 = v20;
  if ( v7 )
  {
    v8 = *(_QWORD *)(a2 + 24) == (_QWORD)v20;
  }
  else
  {
    v14 = (unsigned __int64)v20;
    v15 = v5;
    v8 = sub_C43C50(a2 + 24, &v18);
    v6 = v14;
    v5 = v15;
  }
  if ( v5 > 0x40 )
  {
    if ( v6 )
    {
      v16 = v8;
      j_j___libc_free_0_0(v6);
      v8 = v16;
      if ( v21 > 0x40 )
      {
        if ( v20 )
        {
          j_j___libc_free_0_0((unsigned __int64)v20);
          v8 = v16;
        }
      }
    }
  }
  if ( !v8 )
    return 0;
  v21 = *(_DWORD *)(a2 + 16);
  if ( v21 > 0x40 )
    sub_C43780((__int64)&v20, v4);
  else
    v20 = *(const void **)(a2 + 8);
  sub_C46A40((__int64)&v20, 1);
  v9 = v21;
  v10 = (unsigned __int64)v20;
  v21 = 0;
  v7 = *(_DWORD *)(a2 + 32) <= 0x40u;
  v19 = v9;
  v18 = v20;
  if ( v7 )
  {
    v12 = *(_QWORD *)(a2 + 24) == (_QWORD)v20;
  }
  else
  {
    v17 = v9;
    v11 = sub_C43C50(a2 + 24, &v18);
    v9 = v17;
    v12 = v11;
  }
  if ( v9 > 0x40 )
  {
    if ( v10 )
    {
      j_j___libc_free_0_0(v10);
      if ( v21 > 0x40 )
      {
        if ( v20 )
          j_j___libc_free_0_0((unsigned __int64)v20);
      }
    }
  }
  if ( !v12 )
    v4 = 0;
  return sub_AD8D80(a3, (__int64)v4);
}
