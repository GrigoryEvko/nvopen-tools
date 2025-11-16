// Function: sub_17D4880
// Address: 0x17d4880
//
__int64 __fastcall sub_17D4880(__int64 a1, const char *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r12

  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(unsigned int *)(v4 + 156);
  if ( !(_DWORD)v5 )
    return 0;
  if ( !*(_BYTE *)(a1 + 489) )
    return sub_15A06D0(*(__int64 ***)(v4 + 184), (__int64)a2, v5, a4);
  v5 = *((unsigned __int8 *)a2 + 16);
  v6 = (__int64)a2;
  if ( (unsigned __int8)v5 <= 0x10u )
    return sub_15A06D0(*(__int64 ***)(v4 + 184), (__int64)a2, v5, a4);
  if ( (unsigned __int8)v5 > 0x17u && (*((_QWORD *)a2 + 6) || *((__int16 *)a2 + 9) < 0) )
  {
    a2 = "nosanitize";
    if ( sub_1625940(v6, "nosanitize", 0xAu) )
    {
      v4 = *(_QWORD *)(a1 + 8);
      return sub_15A06D0(*(__int64 ***)(v4 + 184), (__int64)a2, v5, a4);
    }
  }
  return *sub_17D46A0(a1 + 384, v6);
}
