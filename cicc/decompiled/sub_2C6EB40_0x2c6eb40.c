// Function: sub_2C6EB40
// Address: 0x2c6eb40
//
_QWORD *__fastcall sub_2C6EB40(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  char *v5; // rdx
  char v6; // al
  __int64 v7; // rax
  __int64 v8; // rdx
  unsigned int v9; // eax
  __int64 v10; // r12
  unsigned int v12; // eax
  __int64 v13; // rax
  __int64 v14; // rdx

  v4 = *(_QWORD *)(*(_QWORD *)a2 + 16LL);
  if ( v4 )
  {
    while ( 1 )
    {
      v5 = *(char **)(v4 + 24);
      v6 = *v5;
      if ( (unsigned __int8)(*v5 - 30) <= 0xAu )
        break;
      v4 = *(_QWORD *)(v4 + 8);
      if ( !v4 )
        goto LABEL_18;
    }
    while ( 1 )
    {
      if ( (unsigned __int8)(v6 - 30) > 0xAu )
        goto LABEL_15;
      v7 = *((_QWORD *)v5 + 5);
      if ( v7 )
      {
        v8 = (unsigned int)(*(_DWORD *)(v7 + 44) + 1);
        v9 = *(_DWORD *)(v7 + 44) + 1;
      }
      else
      {
        v8 = 0;
        v9 = 0;
      }
      if ( v9 >= *(_DWORD *)(a3 + 32) )
        goto LABEL_15;
      v10 = *(_QWORD *)(*(_QWORD *)(a3 + 24) + 8 * v8);
      if ( !v10 || a2 == v10 || v10 == *(_QWORD *)(a2 + 8) )
        goto LABEL_15;
      if ( a2 == *(_QWORD *)(v10 + 8) || *(_DWORD *)(v10 + 16) >= *(_DWORD *)(a2 + 16) )
        goto LABEL_19;
      if ( *(_BYTE *)(a3 + 112) )
        break;
      v12 = *(_DWORD *)(a3 + 116) + 1;
      *(_DWORD *)(a3 + 116) = v12;
      if ( v12 > 0x20 )
      {
        sub_B19440(a3);
        if ( *(_DWORD *)(a2 + 72) < *(_DWORD *)(v10 + 72) )
          goto LABEL_19;
        goto LABEL_14;
      }
      v13 = a2;
      do
      {
        v14 = v13;
        v13 = *(_QWORD *)(v13 + 8);
      }
      while ( v13 && *(_DWORD *)(v10 + 16) <= *(_DWORD *)(v13 + 16) );
      if ( v10 != v14 )
        goto LABEL_19;
LABEL_15:
      v4 = *(_QWORD *)(v4 + 8);
      if ( !v4 )
        goto LABEL_18;
      v5 = *(char **)(v4 + 24);
      v6 = *v5;
    }
    if ( *(_DWORD *)(a2 + 72) < *(_DWORD *)(v10 + 72) )
      goto LABEL_19;
LABEL_14:
    if ( *(_DWORD *)(a2 + 76) > *(_DWORD *)(v10 + 76) )
      goto LABEL_19;
    goto LABEL_15;
  }
LABEL_18:
  v10 = 0;
  v4 = 0;
LABEL_19:
  *a1 = v4;
  a1[2] = v10;
  a1[3] = a3;
  a1[1] = a2;
  a1[4] = 0;
  a1[5] = a2;
  a1[6] = 0;
  a1[7] = 0;
  return a1;
}
