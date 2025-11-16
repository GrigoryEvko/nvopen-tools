// Function: sub_1648720
// Address: 0x1648720
//
unsigned __int64 __fastcall sub_1648720(__int64 a1)
{
  _QWORD *v1; // rax
  _QWORD *v2; // rdx

  v1 = sub_1648700(a1);
  if ( (*((_BYTE *)v1 + 23) & 0x40) != 0 )
    v2 = (_QWORD *)*(v1 - 1);
  else
    v2 = &v1[-3 * (*((_DWORD *)v1 + 5) & 0xFFFFFFF)];
  return 0xAAAAAAAAAAAAAAABLL * ((a1 - (__int64)v2) >> 3);
}
