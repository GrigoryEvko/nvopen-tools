// Function: sub_2FE37D0
// Address: 0x2fe37d0
//
__int64 __fastcall sub_2FE37D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  unsigned int v5; // r8d
  unsigned __int16 v6; // ax
  unsigned int v7; // r8d
  __int64 v9; // r13
  unsigned int v10; // edx
  unsigned int v11; // esi

  v4 = sub_2E88D60(a2);
  v6 = *(_WORD *)(a2 + 68);
  if ( v6 == 70 )
  {
    v9 = *(_QWORD *)(v4 + 32);
    v10 = sub_DFE5B0(a3);
    v11 = *(_DWORD *)(*(_QWORD *)(a2 + 32) + 8LL);
    if ( v10 == 1 )
      return 1;
    if ( v10 != 2 )
    {
      if ( v10 <= 2 )
        BUG();
      v10 = 1;
    }
    return sub_2EBF080(v9, v11, v10);
  }
  else
  {
    if ( v6 > 0x46u )
    {
      v7 = 1;
      if ( v6 != 81 )
        LOBYTE(v7) = (unsigned __int16)(v6 - 133) <= 1u;
      return v7;
    }
    LOBYTE(v5) = v6 == 69;
    return v5;
  }
}
