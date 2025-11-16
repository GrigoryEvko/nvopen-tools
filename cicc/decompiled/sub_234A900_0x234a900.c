// Function: sub_234A900
// Address: 0x234a900
//
void __fastcall sub_234A900(__int64 a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // r12

  v1 = *(_QWORD **)(a1 + 8);
  v2 = *(_QWORD **)a1;
  if ( v1 != *(_QWORD **)a1 )
  {
    do
    {
      if ( *v2 )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v2 + 8LL))(*v2);
      ++v2;
    }
    while ( v1 != v2 );
    v2 = *(_QWORD **)a1;
  }
  if ( v2 )
    j_j___libc_free_0((unsigned __int64)v2);
}
