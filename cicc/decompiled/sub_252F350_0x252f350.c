// Function: sub_252F350
// Address: 0x252f350
//
__int64 __fastcall sub_252F350(__int64 a1, __int64 a2, char a3)
{
  __int64 result; // rax
  _BYTE **v5; // r12
  unsigned __int8 *v6; // rdi
  int v7; // ecx
  char v8; // cl
  __int64 v9; // rdi
  unsigned __int8 *v10; // rdx
  bool v11; // al
  char v12; // [rsp-34h] [rbp-34h]
  __int64 v13; // [rsp-20h] [rbp-20h] BYREF

  result = 1;
  if ( (*(_BYTE *)(a2 + 96) & 4) == 0 )
    return result;
  v5 = *(_BYTE ***)a1;
  v6 = *(unsigned __int8 **)(a2 + 16);
  if ( *(_BYTE *)(a2 + 24) != 1 || !v6 )
    goto LABEL_6;
  v7 = *v6;
  if ( (unsigned int)(v7 - 12) <= 1 )
    goto LABEL_7;
  if ( (unsigned __int8)v7 <= 0x15u && (v12 = a3, v11 = sub_AC30F0((__int64)v6), a3 = v12, v11) )
    *v5[1] = v12 ^ 1;
  else
LABEL_6:
    **v5 = 0;
LABEL_7:
  v8 = **(_BYTE **)(a1 + 8);
  if ( v8 == 1 && !a3 )
  {
    result = **(unsigned __int8 **)(a1 + 16);
    if ( !(_BYTE)result )
    {
      v10 = *(unsigned __int8 **)(a2 + 16);
      if ( !v10 || (unsigned int)*v10 - 12 > 1 || **(_BYTE **)(a1 + 24) )
        return result;
    }
  }
  else if ( **(_BYTE **)(a1 + 24) )
  {
    result = **(unsigned __int8 **)(a1 + 16);
    if ( !(_BYTE)result )
      return result;
  }
  if ( **(_BYTE **)(a2 + 8) == 61 || (result = 0, !v8) )
  {
    v9 = *(_QWORD *)(a1 + 40);
    v13 = *(_QWORD *)(a2 + 8);
    sub_252E900(v9, &v13);
    return 1;
  }
  return result;
}
