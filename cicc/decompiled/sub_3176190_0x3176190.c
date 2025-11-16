// Function: sub_3176190
// Address: 0x3176190
//
_QWORD **__fastcall sub_3176190(__int64 a1, __int64 a2)
{
  char v2; // dl
  _QWORD **v3; // rax
  __int64 v4; // rcx
  _QWORD **v5; // r14
  _QWORD *v6; // r12
  _QWORD **v7; // r15
  _QWORD **result; // rax
  __int64 v9; // r13
  const char *v10; // rax
  __int64 v11; // rdx
  unsigned int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // [rsp+8h] [rbp-38h]

  v2 = *(_BYTE *)(a1 + 468);
  v3 = *(_QWORD ***)(a1 + 448);
  if ( v2 )
    v4 = *(unsigned int *)(a1 + 460);
  else
    v4 = *(unsigned int *)(a1 + 456);
  v5 = &v3[v4];
  if ( v3 == v5 )
  {
LABEL_6:
    result = (_QWORD **)(a1 + 440);
    ++*(_QWORD *)(a1 + 440);
    v14 = a1 + 440;
    if ( v2 )
    {
LABEL_7:
      *(_QWORD *)(a1 + 460) = 0;
      return result;
    }
  }
  else
  {
    while ( 1 )
    {
      v6 = *v3;
      v7 = v3;
      if ( (unsigned __int64)*v3 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v5 == ++v3 )
        goto LABEL_6;
    }
    result = (_QWORD **)(a1 + 440);
    v14 = a1 + 440;
    if ( v5 != v7 )
    {
      do
      {
        v9 = *(_QWORD *)(a1 + 16);
        if ( v9 )
        {
          v10 = sub_BD5D20((__int64)v6);
          a2 = (__int64)v6;
          sub_BBB260(v9, (__int64)v6, (__int64)v10, v11);
        }
        sub_B2E860(v6);
        result = v7 + 1;
        if ( v7 + 1 == v5 )
          break;
        while ( 1 )
        {
          v6 = *result;
          v7 = result;
          if ( (unsigned __int64)*result < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v5 == ++result )
            goto LABEL_14;
        }
      }
      while ( v5 != result );
LABEL_14:
      v2 = *(_BYTE *)(a1 + 468);
    }
    ++*(_QWORD *)(a1 + 440);
    if ( v2 )
      goto LABEL_7;
  }
  v12 = 4 * (*(_DWORD *)(a1 + 460) - *(_DWORD *)(a1 + 464));
  v13 = *(unsigned int *)(a1 + 456);
  if ( v12 < 0x20 )
    v12 = 32;
  if ( (unsigned int)v13 <= v12 )
  {
    result = (_QWORD **)memset(*(void **)(a1 + 448), -1, 8 * v13);
    goto LABEL_7;
  }
  return (_QWORD **)sub_C8C990(v14, a2);
}
