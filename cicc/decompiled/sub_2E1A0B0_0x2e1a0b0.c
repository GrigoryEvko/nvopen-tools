// Function: sub_2E1A0B0
// Address: 0x2e1a0b0
//
void __fastcall sub_2E1A0B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5, unsigned __int64 a6)
{
  unsigned int v6; // r13d
  __int64 v9; // rcx
  unsigned int v10; // r8d
  __int64 v11; // rsi
  void *v12; // rdi
  _QWORD *v13; // rdx
  _QWORD *v14; // rsi

  if ( *(_DWORD *)a1 != (_DWORD)a3 )
  {
    v6 = a3;
    sub_2E19FF0(a1, a2, a3, a4, a5, a6);
    *(_DWORD *)a1 = v6;
    v9 = malloc(216LL * v6);
    if ( !v9 && (216LL * v6 || (v9 = malloc(1u)) == 0) )
      sub_C64F00("Allocation failed", 1u);
    *(_QWORD *)(a1 + 8) = v9;
    if ( v6 )
    {
      v10 = 0;
      while ( 1 )
      {
        v11 = v9 + 216LL * v10;
        if ( v11 )
        {
          v12 = (void *)(v11 + 8);
          v13 = (_QWORD *)(v11 + 8);
          *(_DWORD *)v11 = 0;
          *(_QWORD *)(v11 + 208) = a2;
          v14 = (_QWORD *)(v11 + 136);
          memset(v12, 0, 0xC0u);
          v14[8] = 0;
          do
          {
            *v13 = 0;
            v13 += 2;
            *(v13 - 1) = 0;
          }
          while ( v13 != v14 );
        }
        if ( *(_DWORD *)a1 == ++v10 )
          break;
        v9 = *(_QWORD *)(a1 + 8);
      }
    }
  }
}
