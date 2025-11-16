// Function: sub_3578EF0
// Address: 0x3578ef0
//
void __fastcall sub_3578EF0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 i; // r13
  int v8; // eax

  v6 = *(_QWORD *)(a2 + 56);
  for ( i = *a1; a2 + 48 != v6; v6 = *(_QWORD *)(v6 + 8) )
  {
    v8 = *(_DWORD *)(v6 + 44);
    if ( (v8 & 4) != 0 || (v8 & 8) == 0 )
    {
      if ( (*(_QWORD *)(*(_QWORD *)(v6 + 16) + 24LL) & 0x200LL) != 0 )
        return;
    }
    else if ( sub_2E88A90(v6, 512, 1) )
    {
      return;
    }
    sub_3577FF0(i, v6, a3, a4, a5, a6);
  }
}
