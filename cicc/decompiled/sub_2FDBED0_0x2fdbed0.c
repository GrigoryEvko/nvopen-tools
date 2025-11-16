// Function: sub_2FDBED0
// Address: 0x2fdbed0
//
char __fastcall sub_2FDBED0(__int64 a1, __int64 *a2)
{
  __int64 (*v2)(void); // rax
  char result; // al
  __int64 v4; // rdx

  v2 = *(__int64 (**)(void))(*(_QWORD *)a1 + 128LL);
  if ( (char *)v2 != (char *)sub_2FDBD80 )
    return v2();
  result = sub_2E791F0(a2);
  if ( result )
  {
    v4 = *(_QWORD *)(a2[1] + 656);
    if ( *(_DWORD *)(v4 + 336) == 4 )
      return *(_DWORD *)(v4 + 344) == 6 || *(_DWORD *)(v4 + 344) == 0;
  }
  return result;
}
