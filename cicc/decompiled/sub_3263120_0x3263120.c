// Function: sub_3263120
// Address: 0x3263120
//
bool __fastcall sub_3263120(__int64 a1, __int64 a2, __int64 a3)
{
  __int16 v3; // bx
  __int16 v4; // r14
  unsigned __int64 *v5; // r13
  __int64 v6; // r12
  int v7; // eax
  bool result; // al
  unsigned __int64 *v9; // rax
  unsigned int v10; // eax
  unsigned __int64 v11; // r12
  bool v12; // [rsp+Fh] [rbp-41h]
  unsigned __int64 v13; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v14; // [rsp+18h] [rbp-38h]
  unsigned __int64 v15; // [rsp+20h] [rbp-30h] BYREF
  unsigned int v16; // [rsp+28h] [rbp-28h]

  v3 = *(_WORD *)(*(_QWORD *)a2 + 32LL);
  v4 = *(_WORD *)(*(_QWORD *)a3 + 32LL);
  v5 = (unsigned __int64 *)(*(_QWORD *)(*(_QWORD *)a3 + 96LL) + 24LL);
  v6 = *(_QWORD *)(*(_QWORD *)a2 + 96LL) + 24LL;
  v7 = sub_C49970(v6, v5);
  if ( v7 <= 0 )
  {
    if ( v7 )
    {
      v9 = v5;
      v5 = (unsigned __int64 *)v6;
      v6 = (__int64)v9;
      result = 0;
      if ( (((unsigned __int8)v4 | (unsigned __int8)v3) & 8) != 0 )
        return result;
      goto LABEL_6;
    }
    v6 = (__int64)v5;
  }
  result = 0;
  if ( (((unsigned __int8)v4 | (unsigned __int8)v3) & 8) != 0 )
    return result;
LABEL_6:
  v14 = *(_DWORD *)(v6 + 8);
  if ( v14 > 0x40 )
    sub_C43780((__int64)&v13, (const void **)v6);
  else
    v13 = *(_QWORD *)v6;
  sub_C46B40((__int64)&v13, (__int64 *)v5);
  v10 = v14;
  v11 = v13;
  v14 = 0;
  v16 = v10;
  v15 = v13;
  if ( v10 > 0x40 )
  {
    result = (unsigned int)sub_C44630((__int64)&v15) == 1;
    if ( v11 )
    {
      v12 = result;
      j_j___libc_free_0_0(v11);
      result = v12;
      if ( v14 > 0x40 )
      {
        if ( v13 )
        {
          j_j___libc_free_0_0(v13);
          return v12;
        }
      }
    }
  }
  else
  {
    result = 0;
    if ( v13 )
      return (v13 & (v13 - 1)) == 0;
  }
  return result;
}
