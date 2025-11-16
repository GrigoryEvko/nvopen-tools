// Function: sub_1095CA0
// Address: 0x1095ca0
//
__int64 __fastcall sub_1095CA0(_QWORD *a1)
{
  unsigned __int8 *v1; // rdx

  v1 = (unsigned __int8 *)a1[19];
  if ( v1 == (unsigned __int8 *)(a1[20] + a1[21]) )
    return 0xFFFFFFFFLL;
  else
    return *v1;
}
