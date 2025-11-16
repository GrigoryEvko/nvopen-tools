// Function: sub_72A690
// Address: 0x72a690
//
__int64 __fastcall sub_72A690(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rdi
  char v8; // al
  __int64 result; // rax
  __int64 v10; // rdx

  v5 = *(_QWORD *)(a1 + 128);
  if ( v5 && (unsigned int)sub_8D3A70(v5) )
  {
    if ( a3 && (*(_BYTE *)(a3 + 96) & 0x20) != 0 )
    {
      *(_BYTE *)(a1 + 171) |= 0x20u;
      if ( *(_BYTE *)(a1 + 173) == 10 )
        *(_QWORD *)(a1 + 200) = a3;
      else
        *(_QWORD *)(a1 + 184) = a3;
    }
    if ( a4 && (*(_BYTE *)(a4 + 146) & 8) != 0 )
    {
      *(_BYTE *)(a1 + 171) |= 0x20u;
      if ( *(_BYTE *)(a1 + 173) == 10 )
        *(_QWORD *)(a1 + 200) = a4;
      else
        *(_QWORD *)(a1 + 184) = a4;
    }
  }
  if ( *(_QWORD *)(a2 + 176) )
    *(_QWORD *)(*(_QWORD *)(a2 + 184) + 120LL) = a1;
  else
    *(_QWORD *)(a2 + 176) = a1;
  *(_QWORD *)(a2 + 184) = a1;
  v8 = *(_BYTE *)(a1 + 170);
  if ( (v8 & 0x40) != 0 )
  {
    *(_BYTE *)(a2 + 170) |= 0x40u;
    v8 = *(_BYTE *)(a1 + 170);
  }
  if ( (v8 & 2) != 0 || (result = *(unsigned __int8 *)(a1 + 173), (_BYTE)result == 13) )
  {
    *(_BYTE *)(a2 + 170) |= 2u;
    result = *(unsigned __int8 *)(a1 + 173);
  }
  switch ( (_BYTE)result )
  {
    case 9:
      goto LABEL_15;
    case 0xA:
      if ( (*(_BYTE *)(a1 + 192) & 1) != 0 )
LABEL_15:
        *(_BYTE *)(a2 + 192) |= 1u;
      break;
    case 0xB:
      v10 = *(_QWORD *)(a1 + 176);
      result = *(unsigned __int8 *)(v10 + 173);
      if ( (_BYTE)result == 9 || (_BYTE)result == 10 && (*(_BYTE *)(v10 + 192) & 1) != 0 )
        goto LABEL_15;
      break;
  }
  return result;
}
