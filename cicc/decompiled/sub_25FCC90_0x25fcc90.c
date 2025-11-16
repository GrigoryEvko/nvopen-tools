// Function: sub_25FCC90
// Address: 0x25fcc90
//
unsigned __int64 __fastcall sub_25FCC90(__int64 a1, __int64 **a2)
{
  int v2; // ebx
  unsigned __int64 v3; // r12
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // ecx
  __int64 v8; // r8
  unsigned int *v9; // r10
  int v10; // r13d
  unsigned int *v11; // rbx
  __int64 *v12; // r15
  __int64 v13; // rax
  bool v14; // of
  __int64 *v16; // [rsp+8h] [rbp-48h]
  __int64 *v17; // [rsp+10h] [rbp-40h]
  unsigned int *v18; // [rsp+18h] [rbp-38h]

  v16 = a2[1];
  if ( *a2 == v16 )
    return 0;
  v17 = *a2;
  v2 = 0;
  v3 = 0;
  do
  {
    v4 = *v17;
    v5 = (*(__int64 (__fastcall **)(_QWORD, _QWORD))(a1 + 40))(
           *(_QWORD *)(a1 + 48),
           *(_QWORD *)(*(_QWORD *)(*v17 + 272) + 72LL));
    v9 = *(unsigned int **)(v4 + 200);
    v18 = &v9[*(unsigned int *)(v4 + 208)];
    if ( v18 != v9 )
    {
      v10 = v2;
      v11 = *(unsigned int **)(v4 + 200);
      v12 = (__int64 *)v5;
      do
      {
        sub_25F6560((_QWORD *)v4, *v11, v6, v7, v8);
        v13 = sub_DFD4A0(v12);
        v7 = 1;
        if ( (_DWORD)v6 == 1 )
          v10 = 1;
        v14 = __OFADD__(v13, v3);
        v3 += v13;
        if ( v14 )
        {
          v3 = 0x8000000000000000LL;
          if ( v13 > 0 )
            v3 = 0x7FFFFFFFFFFFFFFFLL;
        }
        ++v11;
      }
      while ( v18 != v11 );
      v2 = v10;
    }
    ++v17;
  }
  while ( v16 != v17 );
  return v3;
}
