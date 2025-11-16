// Function: sub_98B210
// Address: 0x98b210
//
__int64 __fastcall sub_98B210(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // r13
  char v5; // al
  __int64 v6; // rdi
  int v7; // r12d
  int i; // ebx
  __int64 v9; // r8
  unsigned int v10; // eax
  __int64 v11; // r12
  char v12; // dl
  __int64 v13; // rax
  _QWORD *v14; // rbx
  __int64 v15; // rax
  __int64 *v17; // rax
  __int64 v18; // rcx
  __int64 *v19; // rdx
  __int64 v20; // rax
  __int64 v21; // [rsp+0h] [rbp-50h] BYREF
  int v22; // [rsp+8h] [rbp-48h]
  __int64 v23; // [rsp+10h] [rbp-40h]

  v4 = sub_BD3990();
  v5 = *(_BYTE *)v4;
  if ( *(_BYTE *)v4 <= 0x1Cu )
  {
LABEL_4:
    if ( (unsigned __int8)sub_98AE20(v4, &v21, a3, 0) )
    {
      v6 = v21;
      if ( !v21 )
        return 1;
      v7 = v23;
      if ( !(_DWORD)v23 )
        return 1;
      for ( i = 0; ; ++i )
      {
        v9 = sub_AC5320(v6, (unsigned int)(i + v22));
        v10 = i + 1;
        if ( !v9 )
          return v10;
        if ( v7 == v10 )
          break;
        v6 = v21;
      }
      return (unsigned int)(i + 2);
    }
    return 0;
  }
  if ( v5 == 84 )
  {
    if ( *(_BYTE *)(a2 + 28) )
    {
      v17 = *(__int64 **)(a2 + 8);
      v18 = *(unsigned int *)(a2 + 20);
      v19 = &v17[v18];
      if ( v17 != v19 )
      {
        while ( v4 != *v17 )
        {
          if ( v19 == ++v17 )
            goto LABEL_12;
        }
        return -1;
      }
LABEL_12:
      if ( (unsigned int)v18 < *(_DWORD *)(a2 + 16) )
      {
        *(_DWORD *)(a2 + 20) = v18 + 1;
        *v19 = v4;
        ++*(_QWORD *)a2;
        goto LABEL_14;
      }
    }
    sub_C8CC70(a2, v4);
    if ( v12 )
    {
LABEL_14:
      v13 = 4LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF);
      if ( (*(_BYTE *)(v4 + 7) & 0x40) != 0 )
      {
        v14 = *(_QWORD **)(v4 - 8);
        v4 = (__int64)&v14[v13];
      }
      else
      {
        v14 = (_QWORD *)(v4 - v13 * 8);
      }
      v11 = -1;
      if ( (_QWORD *)v4 == v14 )
        return v11;
      while ( 1 )
      {
        v15 = sub_98B210(*v14, a2, a3);
        if ( !v15 )
          break;
        if ( v15 != -1 )
        {
          if ( v15 != v11 && v11 != -1 )
            return 0;
          v11 = v15;
        }
        v14 += 4;
        if ( (_QWORD *)v4 == v14 )
          return v11;
      }
      return 0;
    }
    return -1;
  }
  if ( v5 != 86 )
    goto LABEL_4;
  v11 = sub_98B210(*(_QWORD *)(v4 - 64), a2, a3);
  if ( !v11 )
    return 0;
  v20 = sub_98B210(*(_QWORD *)(v4 - 32), a2, a3);
  if ( !v20 )
    return 0;
  if ( v11 == -1 )
    return v20;
  if ( v11 != v20 && v20 != -1 )
    return 0;
  return v11;
}
