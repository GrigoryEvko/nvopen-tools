// Function: sub_BDA950
// Address: 0xbda950
//
unsigned __int64 __fastcall sub_BDA950(__int64 a1, const char *a2)
{
  unsigned __int64 result; // rax
  const char **v3; // rdx
  const char *v4; // r14
  __int64 v5; // rcx
  const char *v6; // rax
  __int64 v7; // r15
  _BYTE *v8; // rax
  char v9; // dl
  const char *v10; // [rsp+0h] [rbp-50h] BYREF
  char v11; // [rsp+20h] [rbp-30h]
  char v12; // [rsp+21h] [rbp-2Fh]

  result = *((unsigned __int8 *)a2 - 16);
  if ( (result & 2) != 0 )
  {
    v3 = (const char **)*((_QWORD *)a2 - 4);
    v4 = *v3;
    if ( !*v3 )
      goto LABEL_5;
  }
  else
  {
    result = 8LL * (((unsigned __int8)result >> 2) & 0xF);
    v3 = (const char **)&a2[-result - 16];
    v4 = *v3;
    if ( !*v3 )
      goto LABEL_5;
  }
  result = *(unsigned __int8 *)v4;
  if ( (unsigned __int8)result > 0x24u || (v5 = 0x16007FF000LL, !_bittest64(&v5, result)) )
  {
    v12 = 1;
    v6 = "invalid scope";
    goto LABEL_9;
  }
LABEL_5:
  v4 = v3[2];
  if ( !v4 || *v4 == 16 )
    return result;
  v12 = 1;
  v6 = "invalid file";
LABEL_9:
  v7 = *(_QWORD *)a1;
  v10 = v6;
  v11 = 3;
  if ( v7 )
  {
    sub_CA0E80(&v10, v7);
    v8 = *(_BYTE **)(v7 + 32);
    if ( (unsigned __int64)v8 >= *(_QWORD *)(v7 + 24) )
    {
      sub_CB5D20(v7, 10);
    }
    else
    {
      *(_QWORD *)(v7 + 32) = v8 + 1;
      *v8 = 10;
    }
    result = *(_QWORD *)a1;
    v9 = *(_BYTE *)(a1 + 154);
    *(_BYTE *)(a1 + 153) = 1;
    *(_BYTE *)(a1 + 152) |= v9;
    if ( result )
    {
      sub_BD9900((__int64 *)a1, a2);
      return (unsigned __int64)sub_BD9900((__int64 *)a1, v4);
    }
  }
  else
  {
    result = *(unsigned __int8 *)(a1 + 154);
    *(_BYTE *)(a1 + 153) = 1;
    *(_BYTE *)(a1 + 152) |= result;
  }
  return result;
}
