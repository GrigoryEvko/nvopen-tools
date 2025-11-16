// Function: sub_14ACCB0
// Address: 0x14accb0
//
__int64 __fastcall sub_14ACCB0(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v5; // r13
  unsigned __int8 v6; // al
  __int64 v7; // rdi
  int v8; // r12d
  int i; // ebx
  __int64 v10; // r8
  unsigned int v11; // eax
  __int64 v12; // r12
  __int64 *v13; // rax
  char v14; // dl
  __int64 v15; // rax
  _QWORD *v16; // rbx
  __int64 v17; // rax
  __int64 v19; // rax
  __int64 *v20; // rsi
  unsigned int v21; // edi
  __int64 *v22; // rcx
  __int64 v23; // [rsp+0h] [rbp-50h] BYREF
  int v24; // [rsp+8h] [rbp-48h]
  __int64 v25; // [rsp+10h] [rbp-40h]

  v5 = sub_1649C60(a1);
  v6 = *(_BYTE *)(v5 + 16);
  if ( v6 <= 0x17u )
  {
LABEL_4:
    if ( (unsigned __int8)sub_14ACAF0(v5, (__int64)&v23, a3, 0) )
    {
      v7 = v23;
      if ( !v23 )
        return 1;
      v8 = v25;
      if ( !(_DWORD)v25 )
        return 1;
      for ( i = 0; ; ++i )
      {
        v10 = sub_1595A50(v7, (unsigned int)(i + v24));
        v11 = i + 1;
        if ( !v10 )
          return v11;
        if ( v8 == v11 )
          break;
        v7 = v23;
      }
      return (unsigned int)(i + 2);
    }
    return 0;
  }
  if ( v6 == 77 )
  {
    v13 = *(__int64 **)(a2 + 8);
    if ( *(__int64 **)(a2 + 16) == v13 )
    {
      v20 = &v13[*(unsigned int *)(a2 + 28)];
      v21 = *(_DWORD *)(a2 + 28);
      if ( v13 != v20 )
      {
        v22 = 0;
        while ( v5 != *v13 )
        {
          if ( *v13 == -2 )
            v22 = v13;
          if ( v20 == ++v13 )
          {
            if ( !v22 )
              goto LABEL_44;
            *v22 = v5;
            --*(_DWORD *)(a2 + 32);
            ++*(_QWORD *)a2;
            goto LABEL_14;
          }
        }
        return -1;
      }
LABEL_44:
      if ( v21 < *(_DWORD *)(a2 + 24) )
      {
        *(_DWORD *)(a2 + 28) = v21 + 1;
        *v20 = v5;
        ++*(_QWORD *)a2;
        goto LABEL_14;
      }
    }
    sub_16CCBA0(a2, v5);
    if ( v14 )
    {
LABEL_14:
      v15 = 3LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF);
      if ( (*(_BYTE *)(v5 + 23) & 0x40) != 0 )
      {
        v16 = *(_QWORD **)(v5 - 8);
        v5 = (__int64)&v16[v15];
      }
      else
      {
        v16 = (_QWORD *)(v5 - v15 * 8);
      }
      v12 = -1;
      if ( v16 == (_QWORD *)v5 )
        return v12;
      while ( 1 )
      {
        v17 = sub_14ACCB0(*v16, a2, a3);
        if ( !v17 )
          break;
        if ( v17 != -1 )
        {
          if ( v17 != v12 && v12 != -1 )
            return 0;
          v12 = v17;
        }
        v16 += 3;
        if ( (_QWORD *)v5 == v16 )
          return v12;
      }
      return 0;
    }
    return -1;
  }
  if ( v6 != 79 )
    goto LABEL_4;
  v12 = sub_14ACCB0(*(_QWORD *)(v5 - 48), a2, a3);
  if ( !v12 )
    return 0;
  v19 = sub_14ACCB0(*(_QWORD *)(v5 - 24), a2, a3);
  if ( !v19 )
    return 0;
  if ( v12 == -1 )
    return v19;
  if ( v12 != v19 && v19 != -1 )
    return 0;
  return v12;
}
