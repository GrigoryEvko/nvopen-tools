// Function: sub_33CF110
// Address: 0x33cf110
//
__int64 __fastcall sub_33CF110(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 result; // rax

  v6 = *(_QWORD *)(a1 + 720);
  result = *(unsigned int *)(v6 + 648);
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(v6 + 652) )
  {
    sub_C8D5F0(v6 + 640, (const void *)(v6 + 656), result + 1, 8u, a5, a6);
    result = *(unsigned int *)(v6 + 648);
  }
  *(_QWORD *)(*(_QWORD *)(v6 + 640) + 8 * result) = a2;
  ++*(_DWORD *)(v6 + 648);
  return result;
}
