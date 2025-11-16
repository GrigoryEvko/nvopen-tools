// Function: sub_3264D70
// Address: 0x3264d70
//
bool __fastcall sub_3264D70(__int64 a1, __int64 *a2, __int64 *a3)
{
  bool result; // al
  __int64 v4; // rbx
  __int64 v5; // rdx
  __int64 v6; // rsi
  unsigned int v7; // eax
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rdx
  unsigned int v10; // eax
  __int64 v11; // rdi
  unsigned int v12; // r13d
  unsigned __int64 v13; // r14
  bool v14; // cc
  bool v15; // [rsp+Fh] [rbp-51h]
  bool v16; // [rsp+Fh] [rbp-51h]
  const void *v17; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v18; // [rsp+18h] [rbp-48h]
  const void *v19; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v20; // [rsp+28h] [rbp-38h]
  const void *v21; // [rsp+30h] [rbp-30h] BYREF
  unsigned int v22; // [rsp+38h] [rbp-28h]

  result = 1;
  v4 = *a3;
  v5 = *a2;
  if ( !(v4 | *a2) )
    return result;
  result = v5 != 0 && v4 != 0;
  if ( !result )
    return result;
  v6 = *(_QWORD *)(v5 + 96);
  v7 = *(_DWORD *)(v6 + 32);
  v22 = v7;
  if ( v7 <= 0x40 )
  {
    v8 = *(_QWORD *)(v6 + 24);
LABEL_6:
    v9 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v7) & ~v8;
    if ( !v7 )
      v9 = 0;
    v21 = (const void *)v9;
    goto LABEL_9;
  }
  sub_C43780((__int64)&v21, (const void **)(v6 + 24));
  v7 = v22;
  if ( v22 <= 0x40 )
  {
    v8 = (unsigned __int64)v21;
    goto LABEL_6;
  }
  sub_C43D10((__int64)&v21);
LABEL_9:
  sub_C46250((__int64)&v21);
  v10 = v22;
  v22 = 0;
  v20 = v10;
  v19 = v21;
  sub_C46F20((__int64)&v19, 1u);
  v11 = *(_QWORD *)(v4 + 96);
  v12 = v20;
  v20 = 0;
  v13 = (unsigned __int64)v19;
  v14 = *(_DWORD *)(v11 + 32) <= 0x40u;
  v18 = v12;
  v17 = v19;
  if ( v14 )
    result = *(_QWORD *)(v11 + 24) == (_QWORD)v19;
  else
    result = sub_C43C50(v11 + 24, &v17);
  if ( v12 > 0x40 )
  {
    if ( v13 )
    {
      v15 = result;
      j_j___libc_free_0_0(v13);
      result = v15;
      if ( v20 > 0x40 )
      {
        if ( v19 )
        {
          j_j___libc_free_0_0((unsigned __int64)v19);
          result = v15;
        }
      }
    }
  }
  if ( v22 > 0x40 )
  {
    if ( v21 )
    {
      v16 = result;
      j_j___libc_free_0_0((unsigned __int64)v21);
      return v16;
    }
  }
  return result;
}
