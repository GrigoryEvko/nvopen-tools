// Function: sub_97FA10
// Address: 0x97fa10
//
__int64 __fastcall sub_97FA10(__int64 a1, _BYTE *a2, size_t a3, __int64 a4, char a5)
{
  __int64 v5; // rax

  v5 = sub_97F930(a1, a2, a3, a4, a5);
  if ( v5 )
    return *(_QWORD *)(v5 + 16);
  else
    return 0;
}
