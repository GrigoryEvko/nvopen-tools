// Function: sub_770CE0
// Address: 0x770ce0
//
__int64 __fastcall sub_770CE0(__int64 a1)
{
  __int64 v1; // rdx
  char v2; // al
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 8);
  v2 = *(_BYTE *)(v1 + 8);
  if ( v2 == 1 )
  {
    *(_BYTE *)a1 = 2;
    result = *(_QWORD *)(v1 + 32);
    *(_QWORD *)(a1 + 8) = result;
  }
  else if ( v2 == 2 )
  {
    *(_BYTE *)a1 = 59;
    result = *(_QWORD *)(v1 + 32);
    *(_QWORD *)(a1 + 8) = result;
  }
  else
  {
    if ( v2 )
      sub_721090();
    *(_BYTE *)a1 = 6;
    result = *(_QWORD *)(v1 + 32);
    *(_QWORD *)(a1 + 8) = result;
  }
  return result;
}
