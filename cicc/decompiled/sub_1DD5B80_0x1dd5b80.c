// Function: sub_1DD5B80
// Address: 0x1dd5b80
//
__int64 __fastcall sub_1DD5B80(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = *(_QWORD *)(*(_QWORD *)(a2 + 56) + 96LL);
  *(_QWORD *)(result + 8LL * *(unsigned int *)(a2 + 48)) = 0;
  *(_DWORD *)(a2 + 48) = -1;
  return result;
}
