// Function: sub_15CF6C0
// Address: 0x15cf6c0
//
__int64 __fastcall sub_15CF6C0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rbx
  __int64 *v6; // r15
  unsigned __int64 v7; // r14
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 *v11; // r14
  __int64 *i; // rbx
  const void *v13; // r15
  _QWORD *v14; // rcx
  __int64 v15; // rax
  size_t v16; // r8
  _QWORD *v17; // rax
  __int64 v18; // rax
  int v20; // [rsp+8h] [rbp-68h]
  size_t v21; // [rsp+8h] [rbp-68h]
  unsigned __int64 v22; // [rsp+18h] [rbp-58h] BYREF
  __int64 *v23; // [rsp+20h] [rbp-50h] BYREF
  __int64 v24; // [rsp+30h] [rbp-40h]

  v4 = sub_157EBA0(a2);
  if ( v4 )
  {
    v5 = v4;
    v6 = (__int64 *)(a1 + 16);
    LODWORD(v9) = sub_15F4D60(v4);
    *(_QWORD *)a1 = a1 + 16;
    *(_QWORD *)(a1 + 8) = 0x800000000LL;
    v7 = (int)v9;
    v8 = (int)v9;
    LODWORD(v9) = 0;
    v20 = v7;
    if ( v7 > 8 )
    {
      sub_16CD150(a1, a1 + 16, v8, 8);
      v9 = *(unsigned int *)(a1 + 8);
      v6 = (__int64 *)(*(_QWORD *)a1 + 8 * v9);
    }
    if ( (_DWORD)v7 )
    {
      do
      {
        LODWORD(v7) = v7 - 1;
        v10 = sub_15F4DF0(v5, (unsigned int)v7);
        if ( v6 )
          *v6 = v10;
        ++v6;
      }
      while ( (_DWORD)v7 );
      LODWORD(v9) = *(_DWORD *)(a1 + 8);
    }
  }
  else
  {
    v20 = 0;
    *(_QWORD *)a1 = a1 + 16;
    LODWORD(v9) = 0;
    *(_DWORD *)(a1 + 12) = 8;
  }
  *(_DWORD *)(a1 + 8) = v9 + v20;
  if ( a3 )
  {
    sub_15CE790(&v23, (__int64 *)(a3 + 80), a2);
    if ( v24 != *(_QWORD *)(a3 + 88) + 56LL * *(unsigned int *)(a3 + 104) )
    {
      v11 = *(__int64 **)(v24 + 8);
      for ( i = &v11[*(unsigned int *)(v24 + 16)]; i != v11; *(_DWORD *)(a1 + 8) = (__int64)((__int64)v14 + v16 - v15) >> 3 )
      {
        while ( 1 )
        {
          v18 = *v11;
          v22 = *v11 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v18 & 4) == 0 )
            break;
          ++v11;
          sub_15CDD90(a1, &v22);
          if ( i == v11 )
            return a1;
        }
        v13 = (const void *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8));
        v14 = sub_15CF090(*(_QWORD **)a1, (__int64)v13, (__int64 *)&v22);
        v15 = *(_QWORD *)a1;
        v16 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - (_QWORD)v13;
        if ( v13 != (const void *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8)) )
        {
          v21 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - (_QWORD)v13;
          v17 = memmove(v14, v13, v21);
          v16 = v21;
          v14 = v17;
          v15 = *(_QWORD *)a1;
        }
        ++v11;
      }
    }
  }
  return a1;
}
