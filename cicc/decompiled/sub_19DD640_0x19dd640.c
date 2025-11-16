// Function: sub_19DD640
// Address: 0x19dd640
//
bool __fastcall sub_19DD640(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // rax
  __int64 v4; // rdi

  v3 = *a3;
  v4 = *(_QWORD *)(a1 + 8);
  if ( *(_BYTE *)(*a3 + 8) == 16 )
    v3 = **(_QWORD **)(v3 + 16);
  return 8 * (unsigned int)sub_15A9520(v4, *(_DWORD *)(v3 + 8) >> 8) > *(_DWORD *)(*(_QWORD *)a2 + 8LL) >> 8;
}
