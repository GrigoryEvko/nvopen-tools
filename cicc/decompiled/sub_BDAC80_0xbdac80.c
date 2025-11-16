// Function: sub_BDAC80
// Address: 0xbdac80
//
unsigned __int64 __fastcall sub_BDAC80(__int64 a1, __int64 a2)
{
  unsigned __int64 result; // rax
  __int64 v5; // r14
  _BYTE *v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rdi
  __int64 v9; // r14
  _BYTE *v10; // rax
  __int64 v11; // rdx
  _QWORD v12[4]; // [rsp+0h] [rbp-50h] BYREF
  char v13; // [rsp+20h] [rbp-30h]
  char v14; // [rsp+21h] [rbp-2Fh]

  if ( (unsigned __int16)sub_AF18C0(a2) == 18 )
  {
    result = *(_DWORD *)(a2 + 20) & 0x18000000;
    if ( (_DWORD)result != 402653184 )
      return result;
    v9 = *(_QWORD *)a1;
    v14 = 1;
    v12[0] = "has conflicting flags";
    v13 = 3;
    if ( !v9 )
    {
LABEL_11:
      result = *(unsigned __int8 *)(a1 + 154);
      *(_BYTE *)(a1 + 153) = 1;
      *(_BYTE *)(a1 + 152) |= result;
      return result;
    }
    sub_CA0E80(v12, v9);
    v10 = *(_BYTE **)(v9 + 32);
    if ( (unsigned __int64)v10 >= *(_QWORD *)(v9 + 24) )
    {
      sub_CB5D20(v9, 10);
    }
    else
    {
      *(_QWORD *)(v9 + 32) = v10 + 1;
      *v10 = 10;
    }
    v11 = *(_QWORD *)a1;
    result = *(unsigned __int8 *)(a1 + 154);
    *(_BYTE *)(a1 + 153) = 1;
    *(_BYTE *)(a1 + 152) |= result;
    if ( v11 )
      return (unsigned __int64)sub_BD9900((__int64 *)a1, (const char *)a2);
  }
  else
  {
    v5 = *(_QWORD *)a1;
    v14 = 1;
    v12[0] = "invalid tag";
    v13 = 3;
    if ( !v5 )
      goto LABEL_11;
    sub_CA0E80(v12, v5);
    v6 = *(_BYTE **)(v5 + 32);
    if ( (unsigned __int64)v6 >= *(_QWORD *)(v5 + 24) )
    {
      sub_CB5D20(v5, 10);
    }
    else
    {
      *(_QWORD *)(v5 + 32) = v6 + 1;
      *v6 = 10;
    }
    v7 = *(_QWORD *)a1;
    result = *(unsigned __int8 *)(a1 + 154);
    *(_BYTE *)(a1 + 153) = 1;
    *(_BYTE *)(a1 + 152) |= result;
    if ( v7 )
    {
      sub_A62C00((const char *)a2, v7, a1 + 16, *(_QWORD *)(a1 + 8));
      v8 = *(_QWORD *)a1;
      result = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
      if ( result >= *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
      {
        return sub_CB5D20(v8, 10);
      }
      else
      {
        *(_QWORD *)(v8 + 32) = result + 1;
        *(_BYTE *)result = 10;
      }
    }
  }
  return result;
}
