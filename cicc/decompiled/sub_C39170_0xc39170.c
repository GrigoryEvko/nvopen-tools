// Function: sub_C39170
// Address: 0xc39170
//
__int64 __fastcall sub_C39170(__int64 a1)
{
  __int64 result; // rax
  int v2; // r12d
  __int64 v3; // rax

  result = *(_QWORD *)a1;
  if ( *(_DWORD *)(*(_QWORD *)a1 + 16LL) != 1 )
  {
    v2 = *(_DWORD *)(result + 8);
    v3 = sub_C33900(a1);
    return sub_C45DB0(v3, (unsigned int)(v2 - 2));
  }
  return result;
}
