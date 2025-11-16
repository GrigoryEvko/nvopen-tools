// Function: sub_14F4320
// Address: 0x14f4320
//
__int64 __fastcall sub_14F4320(_BYTE *a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // rdi

  result = (unsigned __int8)a1[8];
  if ( (result & 2) != 0 )
    sub_14F42B0(a1, a2, a3);
  if ( (result & 1) != 0 )
  {
    v4 = *(_QWORD *)a1;
    if ( v4 )
      return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v4 + 8LL))(v4);
  }
  return result;
}
