// Function: sub_20FC9D0
// Address: 0x20fc9d0
//
void __fastcall sub_20FC9D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned __int64 a6)
{
  unsigned int v6; // r14d
  __int64 v8; // r12
  unsigned int v9; // esi
  __int64 v10; // rdx
  __int64 v11; // rax

  if ( *(_DWORD *)a1 != (_DWORD)a3 )
  {
    v6 = a3;
    sub_20FC940(a1, a2, a3, a4, a5, a6);
    *(_DWORD *)a1 = v6;
    v8 = malloc(216LL * v6);
    if ( !v8 )
    {
      if ( 216LL * v6 || (v11 = malloc(1u)) == 0 )
      {
        sub_16BD1C0("Allocation failed", 1u);
        v6 = *(_DWORD *)a1;
      }
      else
      {
        v8 = v11;
      }
    }
    *(_QWORD *)(a1 + 8) = v8;
    if ( v6 )
    {
      v9 = 0;
      while ( 1 )
      {
        v10 = v8 + 216LL * v9;
        if ( v10 )
        {
          *(_DWORD *)v10 = 0;
          *(_QWORD *)(v10 + 208) = a2;
          memset((void *)(v10 + 8), 0, 0xC0u);
          *(_QWORD *)(v10 + 200) = 0;
        }
        if ( *(_DWORD *)a1 == ++v9 )
          break;
        v8 = *(_QWORD *)(a1 + 8);
      }
    }
  }
}
