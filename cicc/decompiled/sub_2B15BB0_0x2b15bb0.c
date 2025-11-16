// Function: sub_2B15BB0
// Address: 0x2b15bb0
//
__int64 __fastcall sub_2B15BB0(__int64 **a1, __int64 a2)
{
  __int64 v3; // r14
  __int64 v4; // rdi
  __int64 v5; // rdx
  unsigned int v6; // eax
  __int64 v7; // r12
  __int64 *v8; // rax
  __int64 v9; // r8
  __int64 v10; // r13
  unsigned int v11; // eax
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned int v16; // eax
  __int64 v17; // rax
  int v18; // eax

  v3 = *(_QWORD *)(a2 + 40);
  v4 = (*a1)[415];
  if ( v3 )
  {
    v5 = (unsigned int)(*(_DWORD *)(v3 + 44) + 1);
    v6 = *(_DWORD *)(v3 + 44) + 1;
  }
  else
  {
    v5 = 0;
    v6 = 0;
  }
  if ( v6 >= *(_DWORD *)(v4 + 32) )
    return 0;
  v7 = *(_QWORD *)(*(_QWORD *)(v4 + 24) + 8 * v5);
  if ( !v7 )
    return 0;
  v8 = a1[1];
  v9 = *v8;
  if ( v3 != *(_QWORD *)(*v8 + 40) )
  {
    v10 = *a1[2];
    if ( v10 == v7 )
      return 0;
    if ( !v10 )
      return 1;
    if ( v10 == *(_QWORD *)(v7 + 8) )
      return 0;
    if ( v7 == *(_QWORD *)(v10 + 8) )
    {
LABEL_26:
      v9 = *v8;
      v14 = *(_QWORD *)(*v8 + 40);
      goto LABEL_27;
    }
    if ( *(_DWORD *)(v10 + 16) < *(_DWORD *)(v7 + 16) )
    {
      if ( *(_BYTE *)(v4 + 112) )
      {
        if ( *(_DWORD *)(v7 + 72) >= *(_DWORD *)(v10 + 72) && *(_DWORD *)(v7 + 76) <= *(_DWORD *)(v10 + 76) )
          return 0;
      }
      else
      {
        v11 = *(_DWORD *)(v4 + 116) + 1;
        *(_DWORD *)(v4 + 116) = v11;
        if ( v11 > 0x20 )
        {
          sub_B19440(v4);
          if ( *(_DWORD *)(v7 + 72) >= *(_DWORD *)(v10 + 72) && *(_DWORD *)(v7 + 76) <= *(_DWORD *)(v10 + 76) )
            return 0;
        }
        else
        {
          v12 = v7;
          do
          {
            v13 = v12;
            v12 = *(_QWORD *)(v12 + 8);
          }
          while ( v12 && *(_DWORD *)(v10 + 16) <= *(_DWORD *)(v12 + 16) );
          if ( v10 == v13 )
            return 0;
        }
        v10 = *a1[2];
        if ( v10 == v7 )
          goto LABEL_25;
        if ( !v10 )
          goto LABEL_25;
        v4 = (*a1)[415];
        if ( v7 == *(_QWORD *)(v10 + 8) )
          goto LABEL_25;
      }
    }
    if ( *(_QWORD *)(v7 + 8) != v10 && *(_DWORD *)(v7 + 16) < *(_DWORD *)(v10 + 16) )
    {
      if ( *(_BYTE *)(v4 + 112) )
      {
        if ( *(_DWORD *)(v10 + 72) >= *(_DWORD *)(v7 + 72) && *(_DWORD *)(v10 + 76) <= *(_DWORD *)(v7 + 76) )
        {
LABEL_25:
          v8 = a1[1];
          goto LABEL_26;
        }
      }
      else
      {
        v16 = *(_DWORD *)(v4 + 116) + 1;
        *(_DWORD *)(v4 + 116) = v16;
        if ( v16 > 0x20 )
        {
          sub_B19440(v4);
          if ( *(_DWORD *)(v10 + 72) >= *(_DWORD *)(v7 + 72) && *(_DWORD *)(v10 + 76) <= *(_DWORD *)(v7 + 76) )
          {
            v9 = *a1[1];
            v14 = *(_QWORD *)(v9 + 40);
LABEL_27:
            if ( v3 != v14 )
              return 1;
            goto LABEL_36;
          }
        }
        else
        {
          do
          {
            v17 = v10;
            v10 = *(_QWORD *)(v10 + 8);
          }
          while ( v10 && *(_DWORD *)(v7 + 16) <= *(_DWORD *)(v10 + 16) );
          if ( v7 == v17 )
            goto LABEL_25;
        }
      }
    }
    return 0;
  }
LABEL_36:
  LOBYTE(v18) = sub_B445A0(v9, a2);
  return v18 ^ 1u;
}
