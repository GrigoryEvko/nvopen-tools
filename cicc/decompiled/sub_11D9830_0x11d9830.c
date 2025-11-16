// Function: sub_11D9830
// Address: 0x11d9830
//
__int64 __fastcall sub_11D9830(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 result; // rax
  __int64 v4; // rdx
  __int64 v5; // rdx

  v2 = *(_QWORD *)(a2 + 24);
  if ( *(_BYTE *)v2 == 84 )
    result = sub_11D6F60(
               a1,
               *(_QWORD *)(*(_QWORD *)(v2 - 8)
                         + 32LL * *(unsigned int *)(v2 + 72)
                         + 8LL * (unsigned int)((a2 - *(_QWORD *)(v2 - 8)) >> 5)));
  else
    result = sub_11D6F60(a1, *(_QWORD *)(v2 + 40));
  if ( *(_QWORD *)a2 )
  {
    v4 = *(_QWORD *)(a2 + 8);
    **(_QWORD **)(a2 + 16) = v4;
    if ( v4 )
      *(_QWORD *)(v4 + 16) = *(_QWORD *)(a2 + 16);
  }
  *(_QWORD *)a2 = result;
  if ( result )
  {
    v5 = *(_QWORD *)(result + 16);
    *(_QWORD *)(a2 + 8) = v5;
    if ( v5 )
      *(_QWORD *)(v5 + 16) = a2 + 8;
    *(_QWORD *)(a2 + 16) = result + 16;
    *(_QWORD *)(result + 16) = a2;
  }
  return result;
}
