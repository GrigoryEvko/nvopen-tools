// Function: sub_3244F70
// Address: 0x3244f70
//
char __fastcall sub_3244F70(__int64 a1)
{
  __int64 *v1; // rbx
  __int64 v2; // rax
  __int64 *v3; // r12
  unsigned __int64 v4; // r13
  _QWORD *v5; // rdx

  v1 = *(__int64 **)(a1 + 152);
  v2 = *(unsigned int *)(a1 + 160);
  v3 = &v1[v2];
  if ( v1 != v3 )
  {
    v4 = 0;
    do
    {
      v2 = *v1;
      if ( *(_DWORD *)(*(_QWORD *)(*v1 + 80) + 32LL) != 3 )
      {
        v5 = *(_QWORD **)(v2 + 16);
        if ( !v5 || (*v5 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
          return v2;
        *(_QWORD *)(v2 + 64) = v4;
        v4 += (unsigned int)sub_3244F20((__int64 *)a1, *v1);
      }
      ++v1;
    }
    while ( v3 != v1 );
    LOBYTE(v2) = -1;
    if ( v4 > 0xFFFFFFFF )
    {
      LOBYTE(v2) = sub_31DF690(*(_QWORD *)a1);
      if ( !(_BYTE)v2 )
        sub_C64ED0("The generated debug information is too large for the 32-bit DWARF format.", 1u);
    }
  }
  return v2;
}
