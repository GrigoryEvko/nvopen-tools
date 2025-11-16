// Function: sub_2539310
// Address: 0x2539310
//
__int64 __fastcall sub_2539310(__int64 a1)
{
  unsigned __int8 (*v1)(void); // rax
  int v2; // eax
  unsigned int v3; // r8d

  v1 = *(unsigned __int8 (**)(void))(*(_QWORD *)a1 + 16LL);
  if ( (char *)v1 == (char *)sub_2505DB0 )
  {
    v2 = *(_DWORD *)(a1 + 20);
    if ( !v2 )
      return 1;
  }
  else
  {
    if ( !v1() )
      return 1;
    v2 = *(_DWORD *)(a1 + 20);
  }
  v3 = 0;
  if ( v2 != *(_DWORD *)(a1 + 16) )
    return v3;
  LOBYTE(v3) = *(_BYTE *)(a1 + 81) == *(_BYTE *)(a1 + 80);
  return v3;
}
