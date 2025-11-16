// Function: sub_1ED8E30
// Address: 0x1ed8e30
//
__int64 __fastcall sub_1ED8E30(__int64 a1, __int64 a2)
{
  __int64 *v4; // rax
  __int64 *v6; // rsi
  unsigned int v7; // edi
  __int64 *v8; // rcx

  v4 = *(__int64 **)(a1 + 568);
  if ( *(__int64 **)(a1 + 576) != v4 )
  {
LABEL_2:
    sub_16CCBA0(a1 + 560, a2);
    goto LABEL_3;
  }
  v6 = &v4[*(unsigned int *)(a1 + 588)];
  v7 = *(_DWORD *)(a1 + 588);
  if ( v4 == v6 )
  {
LABEL_12:
    if ( v7 < *(_DWORD *)(a1 + 584) )
    {
      *(_DWORD *)(a1 + 588) = v7 + 1;
      *v6 = a2;
      ++*(_QWORD *)(a1 + 560);
      goto LABEL_3;
    }
    goto LABEL_2;
  }
  v8 = 0;
  while ( a2 != *v4 )
  {
    if ( *v4 == -2 )
      v8 = v4;
    if ( v6 == ++v4 )
    {
      if ( !v8 )
        goto LABEL_12;
      *v8 = a2;
      --*(_DWORD *)(a1 + 592);
      ++*(_QWORD *)(a1 + 560);
      break;
    }
  }
LABEL_3:
  sub_1F10740(*(_QWORD *)(*(_QWORD *)(a1 + 272) + 272LL), a2);
  return sub_1E16240(a2);
}
