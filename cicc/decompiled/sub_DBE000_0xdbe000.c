// Function: sub_DBE000
// Address: 0xdbe000
//
__int64 __fastcall sub_DBE000(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  unsigned int v4; // eax
  unsigned int v5; // eax
  unsigned int v7; // eax

  v3 = sub_DBB9F0(a2, a3, 1u, 0);
  v4 = *(_DWORD *)(v3 + 8);
  *(_DWORD *)(a1 + 8) = v4;
  if ( v4 > 0x40 )
  {
    sub_C43780(a1, (const void **)v3);
    v7 = *(_DWORD *)(v3 + 24);
    *(_DWORD *)(a1 + 24) = v7;
    if ( v7 <= 0x40 )
      goto LABEL_3;
  }
  else
  {
    *(_QWORD *)a1 = *(_QWORD *)v3;
    v5 = *(_DWORD *)(v3 + 24);
    *(_DWORD *)(a1 + 24) = v5;
    if ( v5 <= 0x40 )
    {
LABEL_3:
      *(_QWORD *)(a1 + 16) = *(_QWORD *)(v3 + 16);
      return a1;
    }
  }
  sub_C43780(a1 + 16, (const void **)(v3 + 16));
  return a1;
}
