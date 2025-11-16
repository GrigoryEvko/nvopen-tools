// Function: sub_3741080
// Address: 0x3741080
//
__int64 __fastcall sub_3741080(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 160);
  v2 = *(_QWORD *)(a1 + 40);
  if ( v1 )
  {
    *(_QWORD *)(v2 + 752) = v1;
    *(_QWORD *)(*(_QWORD *)(a1 + 40) + 744LL) = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 752LL) + 24LL);
    v3 = *(_QWORD *)(a1 + 40);
    v4 = *(_QWORD *)(v3 + 752);
    if ( !v4 )
      BUG();
    if ( (*(_BYTE *)v4 & 4) == 0 )
    {
      while ( (*(_BYTE *)(v4 + 44) & 8) != 0 )
        v4 = *(_QWORD *)(v4 + 8);
    }
    result = *(_QWORD *)(v4 + 8);
    *(_QWORD *)(v3 + 752) = result;
  }
  else
  {
    result = sub_2E311E0(*(_QWORD *)(v2 + 744));
    *(_QWORD *)(v2 + 752) = result;
  }
  return result;
}
