// Function: sub_1D91900
// Address: 0x1d91900
//
__int64 __fastcall sub_1D91900(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rdx
  unsigned int v3; // r9d
  int v4; // edi
  __int64 v5; // rax
  unsigned int v6; // esi
  int v7; // ecx
  unsigned int v8; // r8d
  unsigned __int8 v10; // cl

  v2 = *a1;
  v3 = *(_DWORD *)(*a1 + 8);
  v4 = *(_DWORD *)(*a1 + 12);
  if ( v3 == 7 )
    v4 = -(*(_DWORD *)(v2 + 16) + v4);
  v5 = *a2;
  v6 = *(_DWORD *)(*a2 + 8);
  v7 = *(_DWORD *)(v5 + 12);
  if ( v6 == 7 )
    v7 = -(*(_DWORD *)(v5 + 16) + v7);
  v8 = 1;
  if ( v4 <= v7 )
  {
    v8 = 0;
    if ( v4 == v7 )
    {
      v10 = *(_BYTE *)(v2 + 20);
      if ( (v10 & 1) != 0 || (v8 = *(_BYTE *)(v5 + 20) & 1, (*(_BYTE *)(v5 + 20) & 1) == 0) )
      {
        v8 = 0;
        if ( ((*(_BYTE *)(v5 + 20) ^ v10) & 1) == 0 )
        {
          v8 = 1;
          if ( v3 >= v6 )
          {
            v8 = 0;
            if ( v3 == v6 )
              LOBYTE(v8) = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v5 + 16LL) + 48LL) > *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v2 + 16LL)
                                                                                             + 48LL);
          }
        }
      }
    }
  }
  return v8;
}
