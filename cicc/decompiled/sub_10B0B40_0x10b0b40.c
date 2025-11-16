// Function: sub_10B0B40
// Address: 0x10b0b40
//
__int64 __fastcall sub_10B0B40(__int64 a1, unsigned __int8 *a2)
{
  __int64 v2; // rax
  unsigned int v3; // r8d
  unsigned __int8 *v5; // rdx
  unsigned __int8 *v6; // rcx
  int v7; // r9d
  int v8; // eax
  int v9; // esi

  v2 = *((_QWORD *)a2 + 2);
  v3 = 0;
  if ( v2 )
  {
    if ( !*(_QWORD *)(v2 + 8) && *a2 == *(_DWORD *)(a1 + 48) + 29 )
    {
      v5 = (unsigned __int8 *)*((_QWORD *)a2 - 8);
      v6 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
      v7 = *v5;
      v8 = *(_DWORD *)(a1 + 16) + 29;
      v9 = *v6;
      if ( v7 != v8
        || *((_QWORD *)v5 - 8) != **(_QWORD **)a1
        || *((_QWORD *)v5 - 4) != **(_QWORD **)(a1 + 8)
        || *(_DWORD *)(a1 + 40) + 29 != v9
        || *((_QWORD *)v6 - 8) != **(_QWORD **)(a1 + 24)
        || (v3 = 1, *((_QWORD *)v6 - 4) != **(_QWORD **)(a1 + 32)) )
      {
        v3 = 0;
        if ( v8 == v9
          && *((_QWORD *)v6 - 8) == **(_QWORD **)a1
          && *((_QWORD *)v6 - 4) == **(_QWORD **)(a1 + 8)
          && v7 == *(_DWORD *)(a1 + 40) + 29
          && *((_QWORD *)v5 - 8) == **(_QWORD **)(a1 + 24) )
        {
          LOBYTE(v3) = **(_QWORD **)(a1 + 32) == *((_QWORD *)v5 - 4);
        }
      }
    }
  }
  return v3;
}
