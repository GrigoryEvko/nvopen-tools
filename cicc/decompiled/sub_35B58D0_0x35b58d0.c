// Function: sub_35B58D0
// Address: 0x35b58d0
//
__int64 __fastcall sub_35B58D0(__int64 a1)
{
  __int64 v1; // r8
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 976);
  *(_QWORD *)(a1 + 976) = 0;
  if ( v1 )
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v1 + 16LL))(v1);
  return result;
}
