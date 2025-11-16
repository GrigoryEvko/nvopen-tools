// Function: sub_116CAE0
// Address: 0x116cae0
//
__int64 __fastcall sub_116CAE0(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 a4)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v9; // r13
  __int64 v10; // rbx
  _QWORD *v11; // rdi

  v5 = sub_1169800(a2, a3, a4);
  if ( v5 )
  {
    v6 = *(_QWORD *)a2;
    v7 = *(unsigned int *)(a2 + 8);
    *(_QWORD *)(a1 + 16) = v5;
    *(_BYTE *)(a1 + 24) = 1;
    *(_QWORD *)a1 = v6;
    *(_QWORD *)(a1 + 8) = v7;
    return a1;
  }
  else
  {
    v9 = *(_QWORD *)a2;
    v10 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
    if ( *(_QWORD *)a2 != v10 )
    {
      do
      {
        v11 = *(_QWORD **)(v10 - 8);
        v10 -= 8;
        sub_B43D60(v11);
      }
      while ( v9 != v10 );
    }
    *(_BYTE *)(a1 + 24) = 0;
    return a1;
  }
}
