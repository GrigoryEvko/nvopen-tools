// Function: sub_15A69E0
// Address: 0x15a69e0
//
__int64 __fastcall sub_15A69E0(__int64 a1, _BYTE *a2, int a3, int a4, int a5)
{
  if ( a2 && *a2 == 16 )
    LODWORD(a2) = 0;
  return sub_15C06A0(*(_QWORD *)(a1 + 8), (_DWORD)a2, a3, a4, a5, 1, 1);
}
