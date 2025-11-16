// Function: sub_3264760
// Address: 0x3264760
//
bool __fastcall sub_3264760(__int64 a1, __int64 *a2, __int64 *a3)
{
  bool result; // al
  __int64 v4; // rbx
  __int64 v5; // rdx
  __int64 v6; // rsi
  unsigned int v7; // eax
  unsigned __int64 v8; // rdx
  const void *v9; // rdx
  __int64 v10; // rdi
  unsigned int v11; // r12d
  unsigned __int64 v12; // r13
  bool v13; // cc
  bool v14; // [rsp+Fh] [rbp-41h]
  const void *v15; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v16; // [rsp+18h] [rbp-38h]
  const void *v17; // [rsp+20h] [rbp-30h] BYREF
  unsigned int v18; // [rsp+28h] [rbp-28h]

  result = 1;
  v4 = *a2;
  v5 = *a3;
  if ( !(v5 | *a2) )
    return result;
  result = v5 != 0 && v4 != 0;
  if ( !result )
    return result;
  v6 = *(_QWORD *)(v5 + 96);
  v7 = *(_DWORD *)(v6 + 32);
  v18 = v7;
  if ( v7 <= 0x40 )
  {
    v8 = *(_QWORD *)(v6 + 24);
LABEL_6:
    v9 = (const void *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v7) & ~v8);
    if ( !v7 )
      v9 = 0;
    v17 = v9;
    goto LABEL_9;
  }
  sub_C43780((__int64)&v17, (const void **)(v6 + 24));
  v7 = v18;
  if ( v18 <= 0x40 )
  {
    v8 = (unsigned __int64)v17;
    goto LABEL_6;
  }
  sub_C43D10((__int64)&v17);
LABEL_9:
  sub_C46250((__int64)&v17);
  v10 = *(_QWORD *)(v4 + 96);
  v11 = v18;
  v18 = 0;
  v12 = (unsigned __int64)v17;
  v13 = *(_DWORD *)(v10 + 32) <= 0x40u;
  v16 = v11;
  v15 = v17;
  if ( v13 )
    result = *(_QWORD *)(v10 + 24) == (_QWORD)v17;
  else
    result = sub_C43C50(v10 + 24, &v15);
  if ( v11 > 0x40 )
  {
    if ( v12 )
    {
      v14 = result;
      j_j___libc_free_0_0(v12);
      result = v14;
      if ( v18 > 0x40 )
      {
        if ( v17 )
        {
          j_j___libc_free_0_0((unsigned __int64)v17);
          return v14;
        }
      }
    }
  }
  return result;
}
