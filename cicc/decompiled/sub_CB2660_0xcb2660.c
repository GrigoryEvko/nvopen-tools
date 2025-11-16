// Function: sub_CB2660
// Address: 0xcb2660
//
__int64 __fastcall sub_CB2660(__int64 a1, const void *a2, size_t a3, unsigned int a4)
{
  unsigned __int64 v5; // rax
  unsigned int *v7; // r15
  __int64 v8; // r9
  __int64 v9; // rdx
  __int64 v11; // rax
  unsigned __int64 v12; // rcx

  if ( (_BYTE)a4 )
  {
    v5 = *(unsigned int *)(a1 + 40);
    if ( v5 > 1 && ((v7 = (unsigned int *)(*(_QWORD *)(a1 + 32) + 4 * v5 - 8), sub_CB2040(*v7)) || sub_CB2090(*v7)) )
    {
      if ( *(_DWORD *)(*(_QWORD *)(a1 + 32) + 4LL * *(unsigned int *)(a1 + 40) - 4) == 4 )
        sub_CB20A0(a1, 0);
      else
        sub_CB1B10(a1, " ", 1u);
      sub_CB1B10(a1, a2, a3);
      v9 = *(_QWORD *)(a1 + 32);
      if ( *(_DWORD *)(v9 + 4LL * *(unsigned int *)(a1 + 40) - 4) == 4 )
      {
        v11 = (unsigned int)(*(_DWORD *)(a1 + 40) - 1);
        v12 = *(unsigned int *)(a1 + 44);
        *(_DWORD *)(a1 + 40) = v11;
        if ( v11 + 1 > v12 )
        {
          sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v11 + 1, 4u, v11 + 1, v8);
          v9 = *(_QWORD *)(a1 + 32);
          v11 = *(unsigned int *)(a1 + 40);
        }
        *(_DWORD *)(v9 + 4 * v11) = 5;
        ++*(_DWORD *)(a1 + 40);
      }
      *(_QWORD *)(a1 + 104) = 1;
      *(_QWORD *)(a1 + 96) = "\n";
    }
    else
    {
      sub_CB1B10(a1, " ", 1u);
      sub_CB1B10(a1, a2, a3);
    }
  }
  return a4;
}
