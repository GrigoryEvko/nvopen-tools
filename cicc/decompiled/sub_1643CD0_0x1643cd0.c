// Function: sub_1643CD0
// Address: 0x1643cd0
//
__int64 __fastcall sub_1643CD0(__int64 **a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  int v4; // eax
  __int64 v5; // rdx
  __int64 v6; // rax

  v3 = **a1;
  v4 = sub_16D1B30(v3 + 2472, a2, a3);
  if ( v4 == -1 )
    return 0;
  v5 = *(_QWORD *)(v3 + 2472);
  v6 = v5 + 8LL * v4;
  if ( v6 == v5 + 8LL * *(unsigned int *)(v3 + 2480) )
    return 0;
  else
    return *(_QWORD *)(*(_QWORD *)v6 + 8LL);
}
