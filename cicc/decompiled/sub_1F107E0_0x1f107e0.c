// Function: sub_1F107E0
// Address: 0x1f107e0
//
__int64 __fastcall sub_1F107E0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdi
  __int64 result; // rax

  v2 = a1 + 336;
  LODWORD(result) = *(_DWORD *)((*(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL) + 24);
  do
  {
    result = (unsigned int)(result + 8);
    *(_DWORD *)(a2 + 24) = result;
    a2 = *(_QWORD *)(a2 + 8);
  }
  while ( a2 != v2 && (unsigned int)result >= *(_DWORD *)(a2 + 24) );
  return result;
}
