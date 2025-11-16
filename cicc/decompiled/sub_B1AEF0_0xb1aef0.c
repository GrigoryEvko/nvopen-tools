// Function: sub_B1AEF0
// Address: 0xb1aef0
//
void __fastcall sub_B1AEF0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rcx
  unsigned int v4; // eax
  unsigned int v5; // edx
  __int64 v6; // r9
  __int64 v7; // rcx
  unsigned int v8; // eax
  _QWORD *v9; // r8

  if ( a3 )
  {
    v3 = (unsigned int)(*(_DWORD *)(a3 + 44) + 1);
    v4 = *(_DWORD *)(a3 + 44) + 1;
  }
  else
  {
    v3 = 0;
    v4 = 0;
  }
  v5 = *(_DWORD *)(a1 + 32);
  v6 = 0;
  if ( v4 < v5 )
    v6 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v3);
  if ( a2 )
  {
    v7 = (unsigned int)(*(_DWORD *)(a2 + 44) + 1);
    v8 = *(_DWORD *)(a2 + 44) + 1;
  }
  else
  {
    v7 = 0;
    v8 = 0;
  }
  v9 = 0;
  if ( v8 < v5 )
    v9 = *(_QWORD **)(*(_QWORD *)(a1 + 24) + 8 * v7);
  *(_BYTE *)(a1 + 112) = 0;
  sub_B1AE50(v9, v6);
}
