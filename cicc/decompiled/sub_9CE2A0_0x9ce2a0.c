// Function: sub_9CE2A0
// Address: 0x9ce2a0
//
__int64 __fastcall sub_9CE2A0(_BYTE *a1)
{
  __int64 result; // rax
  __int64 v2; // rdi

  result = (unsigned __int8)a1[8];
  if ( (result & 2) != 0 )
    sub_9CE230(a1);
  if ( (result & 1) != 0 )
  {
    v2 = *(_QWORD *)a1;
    if ( v2 )
      return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  }
  return result;
}
