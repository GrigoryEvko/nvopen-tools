// Function: sub_388A6C0
// Address: 0x388a6c0
//
__int64 __fastcall sub_388A6C0(__int64 a1, __int64 a2)
{
  unsigned int v2; // edx
  unsigned int v3; // r8d
  char v4; // al
  __int64 result; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // rdx
  unsigned __int64 v8; // rax
  unsigned int v9; // [rsp+Ch] [rbp-34h]
  unsigned __int64 v10; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v11; // [rsp+18h] [rbp-28h]
  char v12; // [rsp+1Ch] [rbp-24h]
  unsigned __int64 v13; // [rsp+20h] [rbp-20h] BYREF
  unsigned int v14; // [rsp+28h] [rbp-18h]

  v2 = *(_DWORD *)(a1 + 8);
  v3 = *(_DWORD *)(a2 + 8);
  if ( v3 == v2 )
  {
    v4 = *(_BYTE *)(a1 + 12);
    if ( v4 == *(_BYTE *)(a2 + 12) )
    {
      if ( !v4 )
        return sub_16AEA10(a1, a2);
      return sub_16A9900(a1, (unsigned __int64 *)a2);
    }
LABEL_15:
    if ( v4 )
    {
      v7 = 1LL << ((unsigned __int8)v3 - 1);
      v8 = *(_QWORD *)a2;
      if ( v3 > 0x40 )
      {
        if ( (*(_QWORD *)(v8 + 8LL * ((v3 - 1) >> 6)) & v7) == 0 )
          return sub_16A9900(a1, (unsigned __int64 *)a2);
      }
      else if ( (v8 & v7) == 0 )
      {
        return sub_16A9900(a1, (unsigned __int64 *)a2);
      }
      return 1;
    }
    v6 = *(_QWORD *)a1;
    if ( v2 > 0x40 )
      v6 = *(_QWORD *)(v6 + 8LL * ((v2 - 1) >> 6));
    if ( (v6 & (1LL << ((unsigned __int8)v2 - 1))) != 0 )
      return 0xFFFFFFFFLL;
    return sub_16A9900(a1, (unsigned __int64 *)a2);
  }
  if ( v3 >= v2 )
  {
    v4 = *(_BYTE *)(a1 + 12);
    if ( v3 > v2 )
    {
      if ( v4 )
      {
        sub_16A5C50((__int64)&v13, (const void **)a1, v3);
        v12 = 1;
      }
      else
      {
        sub_16A5B10((__int64)&v13, (_DWORD *)a1, v3);
        v12 = 0;
      }
      v11 = v14;
      v10 = v13;
      result = sub_388A6C0(&v10, a2);
      if ( v11 <= 0x40 )
        return result;
LABEL_11:
      if ( v10 )
      {
        v9 = result;
        j_j___libc_free_0_0(v10);
        return v9;
      }
      return result;
    }
    goto LABEL_15;
  }
  if ( *(_BYTE *)(a2 + 12) )
  {
    sub_16A5C50((__int64)&v13, (const void **)a2, v2);
    v12 = 1;
  }
  else
  {
    sub_16A5B10((__int64)&v13, (_DWORD *)a2, v2);
    v12 = 0;
  }
  v11 = v14;
  v10 = v13;
  result = sub_388A6C0(a1, &v10);
  if ( v11 > 0x40 )
    goto LABEL_11;
  return result;
}
