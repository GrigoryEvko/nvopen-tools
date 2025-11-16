// Function: sub_1269DC0
// Address: 0x1269dc0
//
__int64 __fastcall sub_1269DC0(__int64 a1, __int64 a2, _BYTE *a3)
{
  char v4; // bl
  __int64 result; // rax
  __int64 v6; // rdx

  v4 = *(_BYTE *)(a2 + 177);
  if ( v4 == 1 )
  {
    result = *(_QWORD *)(a2 + 184);
    *a3 = 1;
  }
  else
  {
    if ( v4 != 2 )
    {
      result = 0;
      if ( v4 == 4 )
        sub_127B550("Function local static initializer is not supported!");
      goto LABEL_4;
    }
    v6 = *(_QWORD *)(a2 + 184);
    result = 0;
    if ( *(_BYTE *)(v6 + 48) != 2 )
    {
LABEL_4:
      *a3 = v4;
      return result;
    }
    result = *(_QWORD *)(v6 + 56);
    *a3 = 2;
  }
  return result;
}
