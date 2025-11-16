// Function: sub_D016D0
// Address: 0xd016d0
//
bool __fastcall sub_D016D0(__int64 a1, __int64 a2)
{
  bool result; // al
  unsigned int v3; // eax
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // rdx
  unsigned int v6; // r14d
  const void *v7; // r13
  bool v8; // cc
  bool v9; // [rsp+Fh] [rbp-41h]
  const void *v10; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v11; // [rsp+18h] [rbp-38h]
  const void *v12; // [rsp+20h] [rbp-30h] BYREF
  unsigned int v13; // [rsp+28h] [rbp-28h]

  if ( *(_BYTE *)(a1 + 49) != *(_BYTE *)(a2 + 49) )
  {
    if ( *(_DWORD *)(a1 + 32) <= 0x40u )
      return *(_QWORD *)(a1 + 24) == *(_QWORD *)(a2 + 24);
    else
      return sub_C43C50(a1 + 24, (const void **)(a2 + 24));
  }
  v3 = *(_DWORD *)(a2 + 32);
  v13 = v3;
  if ( v3 <= 0x40 )
  {
    v4 = *(_QWORD *)(a2 + 24);
LABEL_8:
    v5 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v3) & ~v4;
    if ( !v3 )
      v5 = 0;
    v12 = (const void *)v5;
    goto LABEL_11;
  }
  sub_C43780((__int64)&v12, (const void **)(a2 + 24));
  v3 = v13;
  if ( v13 <= 0x40 )
  {
    v4 = (unsigned __int64)v12;
    goto LABEL_8;
  }
  sub_C43D10((__int64)&v12);
LABEL_11:
  sub_C46250((__int64)&v12);
  v6 = v13;
  v7 = v12;
  v13 = 0;
  v8 = *(_DWORD *)(a1 + 32) <= 0x40u;
  v11 = v6;
  v10 = v12;
  if ( v8 )
    result = *(_QWORD *)(a1 + 24) == (_QWORD)v12;
  else
    result = sub_C43C50(a1 + 24, &v10);
  if ( v6 > 0x40 )
  {
    if ( v7 )
    {
      v9 = result;
      j_j___libc_free_0_0(v7);
      result = v9;
      if ( v13 > 0x40 )
      {
        if ( v12 )
        {
          j_j___libc_free_0_0(v12);
          return v9;
        }
      }
    }
  }
  return result;
}
