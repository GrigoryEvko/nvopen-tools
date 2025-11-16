// Function: sub_39F1430
// Address: 0x39f1430
//
__int64 __fastcall sub_39F1430(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rax
  unsigned int v3; // edx

  v1 = *(unsigned int *)(a1 + 120);
  if ( (_DWORD)v1 )
  {
    v2 = *(_QWORD *)(*(_QWORD *)(a1 + 112) + 32 * v1 - 32);
    if ( v2 )
    {
      v3 = *(_DWORD *)(*(_QWORD *)(a1 + 264) + 480LL);
      if ( v3 )
      {
        if ( (*(_BYTE *)(v2 + 44) & 2) != 0 && v3 > *(_DWORD *)(v2 + 24) )
          *(_DWORD *)(v2 + 24) = v3;
      }
    }
  }
  sub_39F0A30(a1);
  sub_38D4AB0(a1, 0);
  return sub_38D42B0((__int64 *)a1);
}
