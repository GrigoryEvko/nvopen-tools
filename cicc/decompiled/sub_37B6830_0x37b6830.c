// Function: sub_37B6830
// Address: 0x37b6830
//
__int64 __fastcall sub_37B6830(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdi
  __int64 v5; // r13
  int *v6; // r12
  __int64 v7; // rbx
  __int64 v8; // r14
  int v9; // eax
  __int64 v10; // rdi

  v3 = sub_22077B0(0x28u);
  v4 = *(_QWORD *)(a1 + 24);
  v5 = v3;
  LODWORD(v3) = *(_DWORD *)(a1 + 32);
  *(_QWORD *)(v5 + 16) = 0;
  *(_DWORD *)(v5 + 32) = v3;
  LODWORD(v3) = *(_DWORD *)a1;
  *(_QWORD *)(v5 + 24) = 0;
  *(_DWORD *)v5 = v3;
  *(_QWORD *)(v5 + 8) = a2;
  if ( v4 )
    *(_QWORD *)(v5 + 24) = sub_37B6830(v4, v5);
  v6 = *(int **)(a1 + 16);
  if ( v6 )
  {
    v7 = v5;
    do
    {
      v8 = v7;
      v7 = sub_22077B0(0x28u);
      *(_DWORD *)(v7 + 32) = v6[8];
      v9 = *v6;
      *(_QWORD *)(v7 + 16) = 0;
      *(_DWORD *)v7 = v9;
      *(_QWORD *)(v7 + 24) = 0;
      *(_QWORD *)(v8 + 16) = v7;
      *(_QWORD *)(v7 + 8) = v8;
      v10 = *((_QWORD *)v6 + 3);
      if ( v10 )
        *(_QWORD *)(v7 + 24) = sub_37B6830(v10, v7);
      v6 = (int *)*((_QWORD *)v6 + 2);
    }
    while ( v6 );
  }
  return v5;
}
