// Function: sub_ADDDC0
// Address: 0xadddc0
//
void __fastcall sub_ADDDC0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  int v4; // eax
  __int64 *v5; // rdi
  __int64 v6; // r14
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 *v11; // r13
  __int64 *v12; // rdi
  __int64 v13; // rdi
  int v14; // r12d
  int v15; // [rsp-40h] [rbp-40h] BYREF

  if ( a2 && ((*(_BYTE *)(a2 + 1) & 0x7F) == 2 || *(_DWORD *)(a2 - 8)) )
  {
    v3 = *(unsigned int *)(a1 + 352);
    v4 = v3;
    if ( *(_DWORD *)(a1 + 356) <= (unsigned int)v3 )
    {
      v6 = a1 + 360;
      v11 = (__int64 *)sub_C8D7D0(a1 + 344, a1 + 360, 0, 8, &v15);
      v12 = &v11[*(unsigned int *)(a1 + 352)];
      if ( v12 )
      {
        *v12 = a2;
        sub_B96E90(v12, a2, 1);
      }
      sub_ADDB20(a1 + 344, v11, v7, v8, v9, v10);
      v13 = *(_QWORD *)(a1 + 344);
      v14 = v15;
      if ( v6 != v13 )
        _libc_free(v13, v11);
      ++*(_DWORD *)(a1 + 352);
      *(_QWORD *)(a1 + 344) = v11;
      *(_DWORD *)(a1 + 356) = v14;
    }
    else
    {
      v5 = (__int64 *)(*(_QWORD *)(a1 + 344) + 8 * v3);
      if ( v5 )
      {
        *v5 = a2;
        sub_B96E90(v5, a2, 1);
        v4 = *(_DWORD *)(a1 + 352);
      }
      *(_DWORD *)(a1 + 352) = v4 + 1;
    }
  }
}
