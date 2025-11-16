// Function: sub_328FA70
// Address: 0x328fa70
//
void __fastcall sub_328FA70(__int64 *a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // r12
  unsigned __int64 v7; // r15
  unsigned __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // r9
  unsigned __int64 v12; // r15
  __int64 v13; // r14
  unsigned __int64 i; // rcx
  unsigned int v15; // ebx
  unsigned int v16; // ebx

  v6 = *a1;
  if ( (*a1 & 1) != 0 )
  {
    if ( a2 > 0x39 )
    {
      v7 = *a1;
      v8 = v6 >> 58;
      v9 = sub_22077B0(0x48u);
      v12 = v7 >> 1;
      v13 = v9;
      if ( v9 )
      {
        *(_DWORD *)(v9 + 12) = 6;
        *(_QWORD *)v9 = v9 + 16;
        if ( (unsigned int)(v8 + 63) >> 6 )
          *(_QWORD *)(v9 + 16) = 0;
        *(_DWORD *)(v9 + 8) = (unsigned int)(v8 + 63) >> 6;
        *(_DWORD *)(v9 + 64) = v8;
      }
      if ( v8 )
      {
        for ( i = 0; i != v8; ++i )
        {
          if ( _bittest64((const __int64 *)&v12, i) )
            **(_QWORD **)v9 |= 1LL << i;
        }
      }
      v15 = (a2 + 63) >> 6;
      if ( *(_DWORD *)(v9 + 12) < v15 )
        sub_C8D5F0(v9, (const void *)(v9 + 16), v15, 8u, v10, v11);
      *a1 = v13;
    }
  }
  else
  {
    v16 = (a2 + 63) >> 6;
    if ( *(_DWORD *)(v6 + 12) < v16 )
      sub_C8D5F0(*a1, (const void *)(v6 + 16), v16, 8u, a5, a6);
  }
}
