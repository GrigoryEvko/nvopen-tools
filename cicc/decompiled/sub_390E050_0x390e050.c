// Function: sub_390E050
// Address: 0x390e050
//
void __fastcall sub_390E050(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  unsigned __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  int v7; // r8d
  int v8; // r9d
  __int64 v9; // r14
  unsigned __int64 v10; // r13
  int v11; // eax

  v3 = *(_QWORD *)(a1 + 160);
  if ( v3 && *(_QWORD *)(a1 + 88) )
  {
    v4 = sub_38D4B30(v3);
    v9 = *(_QWORD *)(a1 + 88);
    v10 = v4;
    if ( *(_BYTE *)(v4 + 16) == 1 )
    {
      v10 = *(unsigned int *)(v4 + 72);
      *(_BYTE *)(v9 + 72) = 1;
      *(_BYTE *)(v9 + 248) = 1;
    }
    else
    {
      *(_BYTE *)(v9 + 72) = 1;
      *(_BYTE *)(v9 + 248) = 0;
    }
    *(_DWORD *)(v9 + 80) = *(_DWORD *)a2;
    *(_QWORD *)(v9 + 88) = *(_QWORD *)(a2 + 8);
    sub_390DC20(v9 + 96, a2 + 16, v5, v6, v7, v8);
    v11 = *(_DWORD *)(a2 + 160);
    *(_QWORD *)(v9 + 256) = v10;
    *(_DWORD *)(v9 + 240) = v11;
    *(_QWORD *)(a1 + 88) = 0;
  }
}
