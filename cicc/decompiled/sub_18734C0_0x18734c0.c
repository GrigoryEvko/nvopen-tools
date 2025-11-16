// Function: sub_18734C0
// Address: 0x18734c0
//
__int64 __fastcall sub_18734C0(int *a1, __int64 a2, _QWORD *a3)
{
  __int64 v5; // r14
  __int64 v6; // rax
  int v7; // eax
  __int64 v8; // rdi
  int *v9; // r12
  __int64 v10; // r15
  __int64 v11; // rbx
  __int64 v12; // rax
  int v13; // eax
  __int64 v14; // rdi
  __int64 v16; // rdx
  __int64 j; // rax
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 i; // rax
  __int64 v21; // rax

  v5 = a3[1];
  if ( v5 )
  {
    v6 = *(_QWORD *)(v5 + 8);
    a3[1] = v6;
    if ( v6 )
    {
      if ( v5 == *(_QWORD *)(v6 + 24) )
      {
        *(_QWORD *)(v6 + 24) = 0;
        v19 = *(_QWORD *)(a3[1] + 16LL);
        if ( v19 )
        {
          a3[1] = v19;
          for ( i = *(_QWORD *)(v19 + 24); i; i = *(_QWORD *)(i + 24) )
          {
            a3[1] = i;
            v19 = i;
          }
          v21 = *(_QWORD *)(v19 + 16);
          if ( v21 )
            a3[1] = v21;
        }
      }
      else
      {
        *(_QWORD *)(v6 + 16) = 0;
      }
    }
    else
    {
      *a3 = 0;
    }
  }
  else
  {
    v5 = sub_22077B0(40);
  }
  *(_QWORD *)(v5 + 32) = *((_QWORD *)a1 + 4);
  v7 = *a1;
  *(_QWORD *)(v5 + 16) = 0;
  *(_DWORD *)v5 = v7;
  *(_QWORD *)(v5 + 24) = 0;
  *(_QWORD *)(v5 + 8) = a2;
  v8 = *((_QWORD *)a1 + 3);
  if ( v8 )
    *(_QWORD *)(v5 + 24) = sub_18734C0(v8, v5, a3);
  v9 = (int *)*((_QWORD *)a1 + 2);
  v10 = v5;
  if ( v9 )
  {
    v11 = a3[1];
    if ( !v11 )
      goto LABEL_17;
LABEL_9:
    v12 = *(_QWORD *)(v11 + 8);
    a3[1] = v12;
    if ( v12 )
    {
      if ( v11 == *(_QWORD *)(v12 + 24) )
      {
        *(_QWORD *)(v12 + 24) = 0;
        v16 = *(_QWORD *)(a3[1] + 16LL);
        if ( v16 )
        {
          a3[1] = v16;
          for ( j = *(_QWORD *)(v16 + 24); j; j = *(_QWORD *)(j + 24) )
          {
            a3[1] = j;
            v16 = j;
          }
          v18 = *(_QWORD *)(v16 + 16);
          if ( v18 )
            a3[1] = v18;
        }
      }
      else
      {
        *(_QWORD *)(v12 + 16) = 0;
      }
    }
    else
    {
      *a3 = 0;
    }
    for ( *(_QWORD *)(v11 + 32) = *((_QWORD *)v9 + 4); ; *(_QWORD *)(v11 + 32) = *((_QWORD *)v9 + 4) )
    {
      v13 = *v9;
      *(_QWORD *)(v11 + 16) = 0;
      *(_QWORD *)(v11 + 24) = 0;
      *(_DWORD *)v11 = v13;
      *(_QWORD *)(v10 + 16) = v11;
      *(_QWORD *)(v11 + 8) = v10;
      v14 = *((_QWORD *)v9 + 3);
      if ( v14 )
        *(_QWORD *)(v11 + 24) = sub_18734C0(v14, v11, a3);
      v9 = (int *)*((_QWORD *)v9 + 2);
      if ( !v9 )
        break;
      v10 = v11;
      v11 = a3[1];
      if ( v11 )
        goto LABEL_9;
LABEL_17:
      v11 = sub_22077B0(40);
    }
  }
  return v5;
}
