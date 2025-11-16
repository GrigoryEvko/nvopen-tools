// Function: sub_2F8FA50
// Address: 0x2f8fa50
//
void __fastcall sub_2F8FA50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v7; // edx
  unsigned __int64 v8; // rcx
  __int64 v9; // rax
  _QWORD *v10; // rax
  unsigned __int64 v11; // rdx
  _QWORD *v12; // rax

  if ( !*(_BYTE *)(a1 + 16) )
  {
    v7 = *(_DWORD *)(a1 + 32);
    if ( v7 > 0xA )
    {
      *(_BYTE *)(a1 + 16) = 1;
    }
    else
    {
      v8 = *(unsigned int *)(a1 + 36);
      v9 = v7;
      if ( v7 >= v8 )
      {
        v11 = v7 + 1LL;
        if ( v8 < v9 + 1 )
        {
          sub_C8D5F0(a1 + 24, (const void *)(a1 + 40), v11, 0x10u, a5, a6);
          v9 = *(unsigned int *)(a1 + 32);
        }
        v12 = (_QWORD *)(*(_QWORD *)(a1 + 24) + 16 * v9);
        *v12 = a2;
        v12[1] = a3;
        ++*(_DWORD *)(a1 + 32);
      }
      else
      {
        v10 = (_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL * v7);
        if ( v10 )
        {
          *v10 = a2;
          v10[1] = a3;
          v7 = *(_DWORD *)(a1 + 32);
        }
        *(_DWORD *)(a1 + 32) = v7 + 1;
      }
    }
  }
}
