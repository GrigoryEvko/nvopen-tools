// Function: sub_1D0DF70
// Address: 0x1d0df70
//
void __fastcall sub_1D0DF70(__int64 a1)
{
  __int64 v1; // rdx
  int v2; // eax
  int v3; // ecx
  unsigned int v4; // edx
  unsigned int v5; // eax

  v1 = *(_QWORD *)(a1 + 8);
  if ( !v1 )
    return;
  v2 = *(__int16 *)(v1 + 24);
  if ( *(__int16 *)(v1 + 24) >= 0 )
  {
    if ( (_WORD)v2 == 47 )
    {
      *(_DWORD *)(a1 + 20) = 1;
      return;
    }
LABEL_10:
    *(_DWORD *)(a1 + 20) = 0;
    return;
  }
  v3 = ~v2;
  if ( v2 == -10 || v3 == 21 && **(_BYTE **)(v1 + 40) == 1 )
    goto LABEL_10;
  v4 = *(_DWORD *)(v1 + 60);
  v5 = *(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 16LL) + 8LL) + ((__int64)v3 << 6) + 4);
  *(_DWORD *)(a1 + 16) = 0;
  if ( v5 > v4 )
    v5 = v4;
  *(_DWORD *)(a1 + 20) = v5;
}
