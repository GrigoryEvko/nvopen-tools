// Function: sub_39C7CA0
// Address: 0x39c7ca0
//
__int64 __fastcall sub_39C7CA0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // r12
  __int64 v3; // rcx

  result = *(_QWORD *)(a1 + 80);
  if ( *(_DWORD *)(result + 36) != 3 )
  {
    v2 = sub_396DD80(*(_QWORD *)(a1 + 192));
    if ( *(_BYTE *)(*(_QWORD *)(a1 + 200) + 4502LL) )
      v3 = *(_QWORD *)(*(_QWORD *)(v2 + 96) + 8LL);
    else
      v3 = (*(__int64 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 192) + 256LL) + 704LL))(
             *(_QWORD *)(*(_QWORD *)(a1 + 192) + 256LL),
             *(unsigned int *)(a1 + 600));
    result = sub_39A3E10(a1, a1 + 8, 16, v3, *(_QWORD *)(*(_QWORD *)(v2 + 96) + 8LL));
    *(_QWORD *)(a1 + 608) = result;
  }
  return result;
}
