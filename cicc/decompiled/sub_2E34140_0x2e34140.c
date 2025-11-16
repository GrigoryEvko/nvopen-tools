// Function: sub_2E34140
// Address: 0x2e34140
//
void __fastcall sub_2E34140(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v7; // r12
  int *v8; // rax

  if ( a1 != a2 )
  {
    v6 = *(unsigned int *)(a2 + 120);
    if ( (_DWORD)v6 )
    {
      do
      {
        v7 = **(_QWORD **)(a2 + 112);
        v8 = *(int **)(a2 + 144);
        if ( *(int **)(a2 + 152) == v8 )
          sub_2E321B0(a1, **(_QWORD **)(a2 + 112), v6, a4, a5, a6);
        else
          sub_2E33F80(a1, **(_QWORD **)(a2 + 112), *v8, a4, a5, a6);
        sub_2E33650(a2, v7);
        sub_2E32770(v7, a2, a1);
      }
      while ( *(_DWORD *)(a2 + 120) );
    }
    sub_2E33470(*(unsigned int **)(a1 + 144), *(unsigned int **)(a1 + 152));
  }
}
