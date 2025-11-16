// Function: sub_15CF0D0
// Address: 0x15cf0d0
//
__int64 __fastcall sub_15CF0D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  _QWORD *v4; // r12
  __int64 v5; // r14
  __int64 v6; // rbx
  __int64 v7; // rbx
  __int64 v8; // rax
  int v9; // ebx
  __int64 *v10; // r14
  __int64 *i; // rbx
  const void *v12; // r15
  _QWORD *v13; // rcx
  __int64 v14; // rax
  size_t v15; // r8
  _QWORD *v16; // rax
  __int64 v17; // rax
  size_t v20; // [rsp+8h] [rbp-68h]
  unsigned __int64 v21; // [rsp+18h] [rbp-58h] BYREF
  __int64 *v22; // [rsp+20h] [rbp-50h] BYREF
  __int64 v23; // [rsp+30h] [rbp-40h]

  v3 = *(_QWORD *)(a2 + 8);
  if ( v3 )
  {
    while ( (unsigned __int8)(*(_BYTE *)(sub_1648700(v3) + 16) - 25) > 9u )
    {
      v3 = *(_QWORD *)(v3 + 8);
      if ( !v3 )
        goto LABEL_27;
    }
    v4 = (_QWORD *)(a1 + 16);
    v5 = v3;
    v6 = 0;
    *(_QWORD *)a1 = a1 + 16;
    *(_QWORD *)(a1 + 8) = 0x800000000LL;
    while ( 1 )
    {
      v5 = *(_QWORD *)(v5 + 8);
      if ( !v5 )
        break;
      while ( (unsigned __int8)(*(_BYTE *)(sub_1648700(v5) + 16) - 25) <= 9u )
      {
        v5 = *(_QWORD *)(v5 + 8);
        ++v6;
        if ( !v5 )
          goto LABEL_7;
      }
    }
LABEL_7:
    v7 = v6 + 1;
    if ( v7 > 8 )
    {
      sub_16CD150(a1, a1 + 16, v7, 8);
      v4 = (_QWORD *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8));
    }
    v8 = sub_1648700(v3);
LABEL_12:
    if ( v4 )
      *v4 = *(_QWORD *)(v8 + 40);
    while ( 1 )
    {
      v3 = *(_QWORD *)(v3 + 8);
      if ( !v3 )
        break;
      v8 = sub_1648700(v3);
      if ( (unsigned __int8)(*(_BYTE *)(v8 + 16) - 25) <= 9u )
      {
        ++v4;
        goto LABEL_12;
      }
    }
    v9 = *(_DWORD *)(a1 + 8) + v7;
  }
  else
  {
LABEL_27:
    *(_DWORD *)(a1 + 12) = 8;
    v9 = 0;
    *(_QWORD *)a1 = a1 + 16;
  }
  *(_DWORD *)(a1 + 8) = v9;
  if ( a3 )
  {
    sub_15CE790(&v22, (__int64 *)(a3 + 80), a2);
    if ( v23 != *(_QWORD *)(a3 + 88) + 56LL * *(unsigned int *)(a3 + 104) )
    {
      v10 = *(__int64 **)(v23 + 8);
      for ( i = &v10[*(unsigned int *)(v23 + 16)]; i != v10; *(_DWORD *)(a1 + 8) = (__int64)((__int64)v13 + v15 - v14) >> 3 )
      {
        while ( 1 )
        {
          v17 = *v10;
          v21 = *v10 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v17 & 4) == 0 )
            break;
          ++v10;
          sub_15CDD90(a1, &v21);
          if ( i == v10 )
            return a1;
        }
        v12 = (const void *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8));
        v13 = sub_15CF090(*(_QWORD **)a1, (__int64)v12, (__int64 *)&v21);
        v14 = *(_QWORD *)a1;
        v15 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - (_QWORD)v12;
        if ( v12 != (const void *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8)) )
        {
          v20 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - (_QWORD)v12;
          v16 = memmove(v13, v12, v20);
          v15 = v20;
          v13 = v16;
          v14 = *(_QWORD *)a1;
        }
        ++v10;
      }
    }
  }
  return a1;
}
