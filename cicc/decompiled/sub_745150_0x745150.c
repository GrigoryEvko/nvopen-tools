// Function: sub_745150
// Address: 0x745150
//
__int64 __fastcall sub_745150(char a1, _DWORD *a2, __int64 (__fastcall **a3)(const char *, _QWORD))
{
  __int64 result; // rax

  switch ( a1 )
  {
    case 0:
      return result;
    case 1:
      result = sub_7450F0("visibility(\"hidden\")", a2, a3);
      break;
    case 2:
      result = sub_7450F0("visibility(\"protected\")", a2, a3);
      break;
    case 3:
      result = sub_7450F0("visibility(\"internal\")", a2, a3);
      break;
    case 4:
      result = sub_7450F0("visibility(\"default\")", a2, a3);
      break;
    default:
      sub_721090();
  }
  return result;
}
