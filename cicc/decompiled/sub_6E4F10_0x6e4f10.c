// Function: sub_6E4F10
// Address: 0x6e4f10
//
__int64 __fastcall sub_6E4F10(__int64 a1, __int64 a2, int a3, int a4)
{
  __int64 result; // rax
  __int64 v6; // r13

  if ( a4 )
    result = sub_6E4EE0(a1, a2);
  else
    result = sub_6E4BC0(a1, a2);
  if ( !*(_QWORD *)(a1 + 128) || a3 )
    return result;
  result = *(unsigned __int8 *)(a1 + 16);
  if ( (_BYTE)result == 1 )
  {
    v6 = *(_QWORD *)(a1 + 144);
  }
  else
  {
    if ( (_BYTE)result != 2 )
    {
LABEL_7:
      *(_QWORD *)(a1 + 128) = 0;
      return result;
    }
    v6 = *(_QWORD *)(a1 + 288);
    if ( v6 )
      goto LABEL_11;
    if ( *(_BYTE *)(a1 + 317) != 12 || *(_BYTE *)(a1 + 320) != 1 )
      goto LABEL_7;
    result = sub_72E9A0(a1 + 144);
    v6 = result;
  }
  if ( !v6 )
    goto LABEL_7;
LABEL_11:
  result = *(unsigned __int8 *)(a2 + 16);
  if ( (_BYTE)result == 1 )
  {
    result = *(_QWORD *)(a2 + 144);
  }
  else
  {
    if ( (_BYTE)result != 2 )
      goto LABEL_7;
    result = *(_QWORD *)(a2 + 288);
    if ( !result )
    {
      if ( *(_BYTE *)(a2 + 317) != 12 || *(_BYTE *)(a2 + 320) != 1 )
        goto LABEL_7;
      result = sub_72E9A0(a2 + 144);
    }
  }
  if ( result != v6 )
    goto LABEL_7;
  return result;
}
