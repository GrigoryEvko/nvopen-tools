// Function: sub_27ABD90
// Address: 0x27abd90
//
__int64 __fastcall sub_27ABD90(_DWORD *a1, __int64 a2)
{
  if ( a1[2] == *(_DWORD *)(a2 + 8) )
    return (*(__int64 (__fastcall **)(_DWORD *))(*(_QWORD *)a1 + 16LL))(a1);
  else
    return 0;
}
