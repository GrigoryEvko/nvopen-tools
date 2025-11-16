// Function: sub_30A7A60
// Address: 0x30a7a60
//
__int64 __fastcall sub_30A7A60(__int64 a1)
{
  size_t v1; // rdx
  const void *v2; // r13
  __int64 v3; // rax
  unsigned __int8 v4; // dl
  __int64 v5; // rax
  unsigned __int8 *v6; // rax
  bool v7; // cc
  __int64 result; // rax
  __int64 v9; // [rsp+8h] [rbp-38h]
  __int64 v10[2]; // [rsp+10h] [rbp-30h] BYREF
  __int64 v11; // [rsp+20h] [rbp-20h] BYREF

  if ( sub_B2FC80(a1) )
  {
    sub_B2F930(v10, a1);
    result = sub_B2F650(v10[0], v10[1]);
    if ( (__int64 *)v10[0] != &v11 )
    {
      v9 = result;
      j_j___libc_free_0(v10[0]);
      return v9;
    }
  }
  else
  {
    v1 = 0;
    v2 = off_4CE0088;
    if ( off_4CE0088 )
      v1 = strlen((const char *)off_4CE0088);
    v3 = sub_B91CC0(a1, v2, v1);
    v4 = *(_BYTE *)(v3 - 16);
    if ( (v4 & 2) != 0 )
      v5 = *(_QWORD *)(v3 - 32);
    else
      v5 = v3 - 8LL * ((v4 >> 2) & 0xF) - 16;
    v6 = sub_BD3990(*(unsigned __int8 **)(*(_QWORD *)v5 + 136LL), (__int64)v2);
    v7 = *((_DWORD *)v6 + 8) <= 0x40u;
    result = *((_QWORD *)v6 + 3);
    if ( !v7 )
      return *(_QWORD *)result;
  }
  return result;
}
