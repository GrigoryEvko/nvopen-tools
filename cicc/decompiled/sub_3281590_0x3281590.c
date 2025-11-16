// Function: sub_3281590
// Address: 0x3281590
//
__int64 __fastcall sub_3281590(__int64 a1)
{
  unsigned __int16 v1; // ax
  __int64 v3; // [rsp-8h] [rbp-8h]

  v1 = *(_WORD *)a1;
  if ( !*(_WORD *)a1 )
    return sub_3007240(a1);
  *((_BYTE *)&v3 - 4) = (unsigned __int16)(v1 - 176) <= 0x34u;
  *((_DWORD *)&v3 - 2) = word_4456340[v1 - 1];
  return *(&v3 - 1);
}
