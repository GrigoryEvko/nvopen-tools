// Function: sub_B19ED0
// Address: 0xb19ed0
//
__int64 __fastcall sub_B19ED0(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rdx

  v3 = *(_QWORD *)(a3 + 24);
  if ( *(_BYTE *)v3 != 84 )
  {
    v4 = *(_QWORD *)(v3 + 40);
    return sub_B19C20(a1, a2, v4);
  }
  v4 = *(_QWORD *)(*(_QWORD *)(v3 - 8)
                 + 32LL * *(unsigned int *)(v3 + 72)
                 + 8LL * (unsigned int)((a3 - *(_QWORD *)(v3 - 8)) >> 5));
  if ( a2[1] != *(_QWORD *)(v3 + 40) || *a2 != v4 )
    return sub_B19C20(a1, a2, v4);
  return 1;
}
