// Function: sub_8AEEA0
// Address: 0x8aeea0
//
__int64 __fastcall sub_8AEEA0(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  __int64 result; // rax
  char v6; // al

  result = *(unsigned __int8 *)(a3 + 56);
  if ( (result & 1) != 0 )
  {
    if ( (result & 4) != 0 )
    {
      sub_8AEC90(a1, a3, a4);
      v6 = *(_BYTE *)(a2 + 8);
      if ( v6 != 1 )
      {
LABEL_4:
        if ( v6 != 2 )
        {
          if ( v6 )
            sub_721090();
        }
      }
    }
    else
    {
      v6 = *(_BYTE *)(a2 + 8);
      if ( v6 != 1 )
        goto LABEL_4;
    }
    result = *(_QWORD *)(a3 + 80);
    *(_QWORD *)(a2 + 32) = result;
  }
  return result;
}
