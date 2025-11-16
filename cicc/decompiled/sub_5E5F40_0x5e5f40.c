// Function: sub_5E5F40
// Address: 0x5e5f40
//
__int64 __fastcall sub_5E5F40(__int64 a1, int a2, unsigned int a3)
{
  char *v3; // r14
  __int64 *v4; // r13
  __int64 result; // rax
  __int64 v6; // r8
  __int64 i; // r12
  __int64 v8; // rax
  __int64 v9; // [rsp+0h] [rbp-40h]
  __int64 v10; // [rsp+0h] [rbp-40h]

  v3 = "__device__";
  v4 = (__int64 *)(a1 + 36);
  result = *(unsigned __int8 *)(a1 + 32);
  if ( !a2 )
  {
    if ( (result & 1) != 0 )
    {
      sub_684AA0(7, 3615, a1 + 36);
      result = *(unsigned __int8 *)(a1 + 32);
    }
    v3 = "__host__ __device__";
  }
  v6 = *(_QWORD *)(a1 + 8);
  if ( (result & 8) != 0 )
  {
    if ( v6 && (*(_BYTE *)(v6 + 172) & 1) != 0 )
    {
      if ( (result & 0x18) != 0x18 )
        goto LABEL_9;
LABEL_20:
      v10 = v6;
      sub_684B00(3635, v4);
      LOBYTE(result) = *(_BYTE *)(a1 + 32);
      v6 = v10;
      goto LABEL_9;
    }
    v9 = *(_QWORD *)(a1 + 8);
    result = sub_6849F0(7, 3593, v4, v3);
    v6 = v9;
  }
  if ( !v6 )
    return result;
  LOBYTE(result) = *(_BYTE *)(a1 + 32);
  if ( (result & 0x18) == 0x18 && (*(_BYTE *)(v6 + 172) & 1) != 0 )
    goto LABEL_20;
LABEL_9:
  for ( i = *(_QWORD *)(v6 + 120); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( (result & 0x20) != 0
    || (*(_QWORD *)(v6 + 168) & 0x100008000LL) == 0x8000
    && (v8 = *(_QWORD *)(v6 + 128)) != 0
    && (*(_BYTE *)(v8 + 33) & 3) != 0 )
  {
    sub_6849F0(7, 3594, v4, v3);
  }
  sub_5E5D30(i, v4, a2);
  result = a3;
  if ( a3 )
  {
    if ( (*(_BYTE *)(a1 + 32) & 0x10) != 0 )
      return sub_6849F0(7, 3691, v4, v3);
  }
  return result;
}
