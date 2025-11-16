// Function: sub_24F37B0
// Address: 0x24f37b0
//
__int64 __fastcall sub_24F37B0(__int64 *a1, __int64 a2, __int64 a3)
{
  _QWORD **v3; // rbx
  __int64 v4; // r14
  _QWORD *v5; // r13
  _QWORD **v6; // rbx
  __int64 v7; // r12
  _QWORD *v8; // rdi

  v3 = *(_QWORD ***)a2;
  v4 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
  if ( v4 != *(_QWORD *)a2 )
  {
    do
    {
      v5 = *v3++;
      sub_BD84D0((__int64)v5, *a1);
      sub_B43D60(v5);
    }
    while ( (_QWORD **)v4 != v3 );
  }
  *(_DWORD *)(a2 + 8) = 0;
  v6 = *(_QWORD ***)a3;
  v7 = *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8);
  if ( v7 != *(_QWORD *)a3 )
  {
    do
    {
      v8 = *v6++;
      sub_B43D60(v8);
    }
    while ( (_QWORD **)v7 != v6 );
  }
  *(_DWORD *)(a3 + 8) = 0;
  return a3;
}
