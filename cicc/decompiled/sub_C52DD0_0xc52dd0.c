// Function: sub_C52DD0
// Address: 0xc52dd0
//
_QWORD *__fastcall sub_C52DD0(__int64 a1, __int64 a2, __int64 (__fastcall *a3)(__int64, __int64), __int64 a4)
{
  __int64 v6; // rdx
  int v7; // eax
  _QWORD *result; // rax
  __int64 v9; // rdx
  _QWORD *v10; // r13
  _QWORD *v11; // rbx
  _QWORD *v12; // rax
  _QWORD *v13; // rdx
  __int64 v14; // r13
  _QWORD *v15; // rcx
  _QWORD *v16; // rax
  __int64 v17; // rdx
  _QWORD *v18; // rbx
  _QWORD *v19; // r13
  __int64 v20; // rax
  _QWORD *v21; // rax

  v6 = *(unsigned int *)(a2 + 116);
  v7 = *(_DWORD *)(a2 + 120);
  if ( v7 == (_DWORD)v6 )
  {
    v20 = sub_C52570();
    return (_QWORD *)a3(a4, v20);
  }
  if ( (_DWORD)v6 - v7 == 1 )
  {
    v12 = *(_QWORD **)(a2 + 104);
    if ( !*(_BYTE *)(a2 + 124) )
      v6 = *(unsigned int *)(a2 + 112);
    v13 = &v12[v6];
    v14 = *v12;
    if ( v13 != v12 )
    {
      while ( 1 )
      {
        v14 = *v12;
        v15 = v12;
        if ( *v12 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v13 == ++v12 )
        {
          v14 = v15[1];
          break;
        }
      }
    }
    if ( v14 == sub_C52D90() )
    {
      v16 = *(_QWORD **)(a1 + 288);
      if ( *(_BYTE *)(a1 + 308) )
        v17 = *(unsigned int *)(a1 + 300);
      else
        v17 = *(unsigned int *)(a1 + 296);
      v18 = &v16[v17];
      if ( v16 != v18 )
      {
        while ( 1 )
        {
          v19 = v16;
          if ( *v16 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v18 == ++v16 )
            goto LABEL_34;
        }
        if ( v16 != v18 )
        {
          do
          {
            ((void (__fastcall *)(__int64))a3)(a4);
            v21 = v19 + 1;
            if ( v19 + 1 == v18 )
              break;
            while ( 1 )
            {
              v19 = v21;
              if ( *v21 < 0xFFFFFFFFFFFFFFFELL )
                break;
              if ( v18 == ++v21 )
                goto LABEL_34;
            }
          }
          while ( v18 != v21 );
        }
      }
LABEL_34:
      v20 = sub_C52D90();
      return (_QWORD *)a3(a4, v20);
    }
  }
  result = *(_QWORD **)(a2 + 104);
  if ( *(_BYTE *)(a2 + 124) )
    v9 = *(unsigned int *)(a2 + 116);
  else
    v9 = *(unsigned int *)(a2 + 112);
  v10 = &result[v9];
  if ( result != v10 )
  {
    while ( 1 )
    {
      v11 = result;
      if ( *result < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v10 == ++result )
        return result;
    }
    while ( v10 != v11 )
    {
      ((void (__fastcall *)(__int64))a3)(a4);
      result = v11 + 1;
      if ( v11 + 1 == v10 )
        break;
      while ( 1 )
      {
        v11 = result;
        if ( *result < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v10 == ++result )
          return result;
      }
    }
  }
  return result;
}
