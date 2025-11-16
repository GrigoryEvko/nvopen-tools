// Function: sub_AE4570
// Address: 0xae4570
//
__int64 __fastcall sub_AE4570(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  __int64 v3; // rdi
  int v4; // edx
  __int64 v6; // [rsp+8h] [rbp-18h]

  v2 = sub_AE43F0(a1, a2);
  v3 = sub_BCCE00(*(_QWORD *)a2, v2);
  v4 = *(unsigned __int8 *)(a2 + 8);
  if ( (unsigned int)(v4 - 17) > 1 )
    return v3;
  BYTE4(v6) = (_BYTE)v4 == 18;
  LODWORD(v6) = *(_DWORD *)(a2 + 32);
  return sub_BCE1B0(v3, v6);
}
