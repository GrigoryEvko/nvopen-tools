// Function: sub_2646720
// Address: 0x2646720
//
void __fastcall sub_2646720(__int64 a1, int a2)
{
  unsigned int v2; // eax
  _QWORD *v3; // rax
  __int64 v4; // rdx
  _QWORD *i; // rdx

  if ( !a2 )
  {
    *(_DWORD *)(a1 + 24) = 0;
    goto LABEL_9;
  }
  v2 = sub_AF1560(4 * a2 / 3u + 1);
  *(_DWORD *)(a1 + 24) = v2;
  if ( !v2 )
  {
LABEL_9:
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    return;
  }
  v3 = (_QWORD *)sub_C7D670(8LL * v2, 8);
  v4 = *(unsigned int *)(a1 + 24);
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 8) = v3;
  for ( i = &v3[v4]; i != v3; ++v3 )
  {
    if ( v3 )
      *v3 = -4096;
  }
}
