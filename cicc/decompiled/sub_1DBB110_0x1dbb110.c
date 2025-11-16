// Function: sub_1DBB110
// Address: 0x1dbb110
//
__int64 __fastcall sub_1DBB110(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rcx
  __int64 v4; // rdx
  __int64 v5; // rdi

  sub_1DC3BD0(a1[36], a1[29], a1[34], a1[35], a1 + 37);
  v3 = a1[30];
  v4 = 0;
  v5 = a1[36];
  if ( *(_BYTE *)(v3 + 16) )
    v4 = *(unsigned __int8 *)((*(_QWORD *)(*(_QWORD *)(v3 + 24) + 16LL * (*(_DWORD *)(a2 + 112) & 0x7FFFFFFF))
                             & 0xFFFFFFFFFFFFFFF8LL)
                            + 29);
  sub_1DC6220(v5, a2, v4);
  return sub_1DBAEC0((__int64)a1, a2, 0);
}
