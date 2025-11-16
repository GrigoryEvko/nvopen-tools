// Function: sub_15C7D40
// Address: 0x15c7d40
//
__int64 __fastcall sub_15C7D40(__int64 a1, __int64 a2)
{
  __int64 v2; // rax

  v2 = (*(__int64 (__fastcall **)(__int64, const char *))(*(_QWORD *)a2 + 48LL))(
         a2,
         "Instruction selection used fallback path for ");
  return (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v2 + 136LL))(v2, *(_QWORD *)(a1 + 16));
}
