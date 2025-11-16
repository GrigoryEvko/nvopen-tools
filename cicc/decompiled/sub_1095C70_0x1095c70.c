// Function: sub_1095C70
// Address: 0x1095c70
//
__int64 __fastcall sub_1095C70(_QWORD *a1)
{
  unsigned __int8 *v1; // rdx

  v1 = (unsigned __int8 *)a1[19];
  if ( v1 == (unsigned __int8 *)(a1[20] + a1[21]) )
    return 0xFFFFFFFFLL;
  a1[19] = v1 + 1;
  return *v1;
}
