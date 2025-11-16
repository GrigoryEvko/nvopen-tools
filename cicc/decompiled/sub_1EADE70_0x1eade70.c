// Function: sub_1EADE70
// Address: 0x1eade70
//
__int64 __fastcall sub_1EADE70(__int64 a1, __int64 a2)
{
  __int64 (*v2)(void); // rdx
  __int64 result; // rax

  v2 = *(__int64 (**)(void))(*(_QWORD *)a1 + 152LL);
  if ( (char *)v2 == (char *)sub_1E08720 )
    result = (*(unsigned int (**)(void))(*(_QWORD *)a1 + 144LL))() ^ 1;
  else
    result = v2();
  if ( !(_BYTE)result )
    return (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 144LL))(a1, a2);
  return result;
}
