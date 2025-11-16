// Function: sub_11A1260
// Address: 0x11a1260
//
void *__fastcall sub_11A1260(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  unsigned int v3; // eax
  void *result; // rax
  unsigned int v5; // eax

  v2 = *(_DWORD *)(a2 + 8);
  *(_DWORD *)(a1 + 8) = v2;
  if ( v2 > 0x40 )
  {
    sub_C43780(a1, (const void **)a2);
    v5 = *(_DWORD *)(a2 + 24);
    *(_DWORD *)(a1 + 24) = v5;
    if ( v5 <= 0x40 )
      goto LABEL_3;
  }
  else
  {
    *(_QWORD *)a1 = *(_QWORD *)a2;
    v3 = *(_DWORD *)(a2 + 24);
    *(_DWORD *)(a1 + 24) = v3;
    if ( v3 <= 0x40 )
    {
LABEL_3:
      result = *(void **)(a2 + 16);
      *(_QWORD *)(a1 + 16) = result;
      return result;
    }
  }
  return sub_C43780(a1 + 16, (const void **)(a2 + 16));
}
