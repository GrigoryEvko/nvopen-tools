// Function: sub_DEF2A0
// Address: 0xdef2a0
//
void __fastcall sub_DEF2A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // rax
  _QWORD *v8; // r12
  _QWORD *v9; // rbx
  __int64 v10; // rsi
  __int64 v11; // rax
  int v12; // edx

  if ( (*(_DWORD *)(a1 + 136))++ == -1 )
  {
    if ( *(_DWORD *)(a1 + 16) )
    {
      v7 = *(_QWORD **)(a1 + 8);
      v8 = &v7[3 * *(unsigned int *)(a1 + 24)];
      if ( v7 != v8 )
      {
        while ( 1 )
        {
          v9 = v7;
          if ( *v7 != -4096 && *v7 != -8192 )
            break;
          v7 += 3;
          if ( v8 == v7 )
            return;
        }
        while ( v8 != v9 )
        {
          v10 = v9[2];
          v9 += 3;
          v11 = sub_DEEEC0(*(_QWORD *)(a1 + 112), v10, *(_QWORD *)(a1 + 120), *(_QWORD *)(a1 + 128), a5, a6);
          v12 = *(_DWORD *)(a1 + 136);
          *(v9 - 1) = v11;
          *((_DWORD *)v9 - 4) = v12;
          if ( v9 == v8 )
            break;
          while ( *v9 == -8192 || *v9 == -4096 )
          {
            v9 += 3;
            if ( v8 == v9 )
              return;
          }
        }
      }
    }
  }
}
