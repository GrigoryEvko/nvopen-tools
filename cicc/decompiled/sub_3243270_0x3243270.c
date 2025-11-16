// Function: sub_3243270
// Address: 0x3243270
//
__int64 __fastcall sub_3243270(_BYTE *a1)
{
  __int64 result; // rax

  if ( sub_32420F0((__int64)a1) )
    return (*(__int64 (__fastcall **)(_BYTE *, __int64, _QWORD))(*(_QWORD *)a1 + 8LL))(a1, 159, 0);
  result = (a1[101] >> 1) & 0xF;
  if ( (unsigned __int8)result > 3u )
    return (*(__int64 (__fastcall **)(_BYTE *, __int64, _QWORD))(*(_QWORD *)a1 + 8LL))(a1, 159, 0);
  return result;
}
