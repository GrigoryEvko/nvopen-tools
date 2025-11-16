// Function: sub_1AEA1F0
// Address: 0x1aea1f0
//
void __fastcall sub_1AEA1F0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 *v3; // rax
  __int64 v4; // rax
  __int64 i; // r12
  _QWORD *v6; // rax
  int v7; // r8d
  int v8; // r9d
  _QWORD *v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rax

  if ( (*(_BYTE *)(a2 + 23) & 0x10) != 0 )
  {
    v2 = sub_161E8E0(a2);
    if ( v2 )
    {
      v3 = (__int64 *)sub_16498A0(a2);
      v4 = sub_1629050(v3, v2);
      if ( v4 )
      {
        for ( i = *(_QWORD *)(v4 + 8); i; i = *(_QWORD *)(i + 8) )
        {
          while ( 1 )
          {
            v6 = sub_1648700(i);
            v9 = v6;
            if ( *((_BYTE *)v6 + 16) == 78 )
            {
              v10 = *(v6 - 3);
              if ( !*(_BYTE *)(v10 + 16) && (*(_BYTE *)(v10 + 33) & 0x20) != 0 && *(_DWORD *)(v10 + 36) == 38 )
                break;
            }
            i = *(_QWORD *)(i + 8);
            if ( !i )
              return;
          }
          v11 = *(unsigned int *)(a1 + 8);
          if ( (unsigned int)v11 >= *(_DWORD *)(a1 + 12) )
          {
            sub_16CD150(a1, (const void *)(a1 + 16), 0, 8, v7, v8);
            v11 = *(unsigned int *)(a1 + 8);
          }
          *(_QWORD *)(*(_QWORD *)a1 + 8 * v11) = v9;
          ++*(_DWORD *)(a1 + 8);
        }
      }
    }
  }
}
