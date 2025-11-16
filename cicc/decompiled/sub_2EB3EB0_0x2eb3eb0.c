// Function: sub_2EB3EB0
// Address: 0x2eb3eb0
//
void __fastcall sub_2EB3EB0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // ecx
  __int64 v4; // rcx
  unsigned int v5; // eax
  unsigned int v6; // edx
  __int64 v7; // r8
  __int64 v8; // rcx
  unsigned int v9; // eax
  __int64 v10; // rsi
  __int64 v11; // rdx

  if ( a2 != a3 )
  {
    if ( a3 )
    {
      v4 = (unsigned int)(*(_DWORD *)(a3 + 24) + 1);
      v5 = *(_DWORD *)(a3 + 24) + 1;
    }
    else
    {
      v4 = 0;
      v5 = 0;
    }
    v6 = *(_DWORD *)(a1 + 56);
    v7 = 0;
    if ( v5 < v6 )
      v7 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8 * v4);
    if ( a2 )
    {
      v8 = (unsigned int)(*(_DWORD *)(a2 + 24) + 1);
      v9 = *(_DWORD *)(a2 + 24) + 1;
    }
    else
    {
      v8 = 0;
      v9 = 0;
    }
    v10 = 0;
    if ( v9 < v6 )
      v10 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8 * v8);
    v11 = v7;
    if ( v7 != v10
      && v7 != 0
      && v10
      && v10 != *(_QWORD *)(v7 + 8)
      && v7 != *(_QWORD *)(v10 + 8)
      && *(_DWORD *)(v10 + 16) < *(_DWORD *)(v7 + 16) )
    {
      if ( *(_BYTE *)(a1 + 136) )
      {
        if ( *(_DWORD *)(v10 + 72) <= *(_DWORD *)(v7 + 72) && *(_DWORD *)(v7 + 76) > *(_DWORD *)(v10 + 76) )
          nullsub_2026();
      }
      else
      {
        v3 = *(_DWORD *)(a1 + 140) + 1;
        *(_DWORD *)(a1 + 140) = v3;
        if ( v3 > 0x20 )
        {
          sub_2EB3E68(a1, v10, v7);
        }
        else
        {
          do
            v11 = *(_QWORD *)(v11 + 8);
          while ( v11 && *(_DWORD *)(v10 + 16) <= *(_DWORD *)(v11 + 16) );
        }
      }
    }
  }
}
