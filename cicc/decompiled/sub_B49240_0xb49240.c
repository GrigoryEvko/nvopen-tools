// Function: sub_B49240
// Address: 0xb49240
//
__int64 __fastcall sub_B49240(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 - 32);
  if ( v1 && !*(_BYTE *)v1 && *(_QWORD *)(v1 + 24) == *(_QWORD *)(a1 + 80) )
    return *(unsigned int *)(v1 + 36);
  else
    return 0;
}
