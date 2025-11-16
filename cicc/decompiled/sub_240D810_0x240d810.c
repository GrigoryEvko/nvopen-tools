// Function: sub_240D810
// Address: 0x240d810
//
__int64 __fastcall sub_240D810(__int64 *a1, unsigned __int8 *a2)
{
  unsigned __int8 *v2; // rax
  unsigned int v3; // r8d
  __int64 v5; // rbx
  const char *v6; // r14
  size_t v7; // rdx
  size_t v8; // r13
  __int64 v9; // r12
  int v10; // eax
  int v11; // eax
  __int64 v12; // rax

  v2 = sub_BD3990(a2, (__int64)a2);
  v3 = 0;
  if ( *v2 == 3 )
  {
    v3 = v2[80] & 1;
    if ( (v2[80] & 1) != 0 )
    {
      v3 = 0;
      if ( (v2[7] & 0x10) != 0 )
      {
        v5 = *a1;
        v6 = sub_BD5D20((__int64)v2);
        v8 = v7;
        v9 = *(_QWORD *)(v5 + 896) + 8LL * *(unsigned int *)(v5 + 904);
        v10 = sub_C92610();
        v11 = sub_C92860((__int64 *)(v5 + 896), v6, v8, v10);
        if ( v11 == -1 )
          v12 = *(_QWORD *)(v5 + 896) + 8LL * *(unsigned int *)(v5 + 904);
        else
          v12 = *(_QWORD *)(v5 + 896) + 8LL * v11;
        LOBYTE(v3) = v12 != v9;
      }
    }
  }
  return v3;
}
