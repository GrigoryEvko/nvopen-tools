// Function: sub_1209B90
// Address: 0x1209b90
//
__int64 __fastcall sub_1209B90(__int64 a1, const void *a2, unsigned __int64 a3)
{
  size_t v3; // r12
  int v4; // eax
  size_t v6; // rdx
  int v7; // eax
  int v8; // eax
  __int64 v9; // rax

  v3 = a3;
  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 >= 0 )
  {
    v6 = v4;
    if ( v4 < v3 )
    {
      v3 = 1;
      if ( v4 > 1 )
      {
        if ( a3 <= v4 )
          v6 = a3;
        v3 = v6;
      }
    }
  }
  v7 = sub_C92610();
  v8 = sub_C92860((__int64 *)a1, a2, v3, v7);
  if ( v8 == -1 )
    return 0;
  v9 = *(_QWORD *)a1 + 8LL * v8;
  if ( v9 == *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) )
    return 0;
  else
    return *(_QWORD *)(*(_QWORD *)v9 + 8LL);
}
