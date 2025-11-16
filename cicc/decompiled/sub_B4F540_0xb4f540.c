// Function: sub_B4F540
// Address: 0xb4f540
//
__int64 __fastcall sub_B4F540(__int64 a1)
{
  __int64 v1; // rax
  int v2; // ebx
  int v3; // r12d
  int *v4; // r13
  unsigned int v5; // r8d
  int *v6; // rax

  v1 = *(_QWORD *)(a1 + 8);
  if ( *(_BYTE *)(v1 + 8) == 18 )
    return 0;
  v2 = *(_DWORD *)(v1 + 32);
  v3 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 - 64) + 8LL) + 32LL);
  if ( v3 < v2 )
  {
    v4 = *(int **)(a1 + 72);
    v5 = sub_B487F0(v4, *(unsigned int *)(a1 + 80), v3);
    if ( (_BYTE)v5 )
    {
      v6 = &v4[v3];
      while ( *v6 == -1 )
      {
        if ( ++v6 == &v4[v3 + 1 + (unsigned __int64)(unsigned int)(v2 - 1 - v3)] )
          return v5;
      }
    }
  }
  return 0;
}
