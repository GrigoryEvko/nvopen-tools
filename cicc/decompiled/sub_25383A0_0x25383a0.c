// Function: sub_25383A0
// Address: 0x25383a0
//
__int64 __fastcall sub_25383A0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdi
  __int64 v5; // r13
  __int64 v6; // rax
  int *v7; // r12
  __int64 v8; // rbx
  __int64 v9; // r14
  int v10; // eax
  __int64 v11; // rdi

  v3 = sub_22077B0(0x28u);
  v4 = *(_QWORD *)(a1 + 24);
  v5 = v3;
  v6 = *(_QWORD *)(a1 + 32);
  *(_QWORD *)(v5 + 16) = 0;
  *(_QWORD *)(v5 + 32) = v6;
  LODWORD(v6) = *(_DWORD *)a1;
  *(_QWORD *)(v5 + 24) = 0;
  *(_DWORD *)v5 = v6;
  *(_QWORD *)(v5 + 8) = a2;
  if ( v4 )
    *(_QWORD *)(v5 + 24) = sub_25383A0(v4, v5);
  v7 = *(int **)(a1 + 16);
  if ( v7 )
  {
    v8 = v5;
    do
    {
      v9 = v8;
      v8 = sub_22077B0(0x28u);
      *(_QWORD *)(v8 + 32) = *((_QWORD *)v7 + 4);
      v10 = *v7;
      *(_QWORD *)(v8 + 16) = 0;
      *(_DWORD *)v8 = v10;
      *(_QWORD *)(v8 + 24) = 0;
      *(_QWORD *)(v9 + 16) = v8;
      *(_QWORD *)(v8 + 8) = v9;
      v11 = *((_QWORD *)v7 + 3);
      if ( v11 )
        *(_QWORD *)(v8 + 24) = sub_25383A0(v11, v8);
      v7 = (int *)*((_QWORD *)v7 + 2);
    }
    while ( v7 );
  }
  return v5;
}
