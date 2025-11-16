// Function: sub_1632000
// Address: 0x1632000
//
__int64 __fastcall sub_1632000(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  int v4; // eax
  __int64 v5; // rax

  v3 = *(_QWORD *)(a1 + 120);
  v4 = sub_16D1B30(v3, a2, a3);
  if ( v4 == -1 )
    return 0;
  v5 = *(_QWORD *)v3 + 8LL * v4;
  if ( v5 == *(_QWORD *)v3 + 8LL * *(unsigned int *)(v3 + 8) )
    return 0;
  else
    return *(_QWORD *)(*(_QWORD *)v5 + 8LL);
}
