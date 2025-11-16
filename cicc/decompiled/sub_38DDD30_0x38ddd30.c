// Function: sub_38DDD30
// Address: 0x38ddd30
//
__int64 __fastcall sub_38DDD30(__int64 a1, unsigned int *a2)
{
  __int64 (__fastcall *v2)(__int64, unsigned int *); // rax

  v2 = *(__int64 (__fastcall **)(__int64, unsigned int *))(*(_QWORD *)a1 + 416LL);
  if ( v2 == sub_38DDC00 )
    return sub_38DDAF0(a1, a2);
  else
    return v2(a1, a2);
}
