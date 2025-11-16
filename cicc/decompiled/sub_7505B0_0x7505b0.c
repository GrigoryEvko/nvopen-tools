// Function: sub_7505B0
// Address: 0x7505b0
//
__int64 __fastcall sub_7505B0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rax

  if ( !*(_DWORD *)(a1 + 56) )
    return (*(__int64 (__fastcall **)(char *))a2)("this");
  v3 = sub_750540(a1, *(__int64 **)(a2 + 128));
  return (*(__int64 (__fastcall **)(_QWORD, __int64))a2)(v3[3], a2);
}
