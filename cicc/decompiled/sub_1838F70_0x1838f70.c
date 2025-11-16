// Function: sub_1838F70
// Address: 0x1838f70
//
void __fastcall sub_1838F70(unsigned __int64 *a1)
{
  unsigned __int64 v2; // r12
  __int64 i; // r14
  __int64 v4; // rbx
  _QWORD *v5; // rax
  __int64 v6; // rdi
  __int64 v7; // rbx
  __int64 *v8; // rax
  char v9; // dl
  __int64 *v10; // rcx
  unsigned int v11; // r8d
  __int64 *v12; // rsi
  __int64 v13; // [rsp+0h] [rbp-40h] BYREF
  char v14; // [rsp+10h] [rbp-30h]

  v2 = a1[2];
  if ( !*(_BYTE *)(v2 - 8) )
    goto LABEL_25;
  while ( 1 )
  {
    while ( 1 )
    {
LABEL_2:
      for ( i = *(_QWORD *)(v2 - 16); !i; *(_QWORD *)(v2 - 16) = i )
      {
        a1[2] -= 24LL;
        v2 = a1[2];
        if ( v2 == a1[1] )
          return;
        if ( *(_BYTE *)(v2 - 8) )
          goto LABEL_2;
LABEL_25:
        for ( i = *(_QWORD *)(*(_QWORD *)(v2 - 24) + 8LL); i; i = *(_QWORD *)(i + 8) )
        {
          if ( (unsigned __int8)(*((_BYTE *)sub_1648700(i) + 16) - 25) <= 9u )
            break;
        }
        *(_BYTE *)(v2 - 8) = 1;
      }
      v4 = *(_QWORD *)(i + 8);
      for ( *(_QWORD *)(v2 - 16) = v4; v4; *(_QWORD *)(v2 - 16) = v4 )
      {
        if ( (unsigned __int8)(*((_BYTE *)sub_1648700(v4) + 16) - 25) <= 9u )
          break;
        v4 = *(_QWORD *)(v4 + 8);
      }
      v5 = sub_1648700(i);
      v6 = *a1;
      v7 = v5[5];
      v8 = *(__int64 **)(*a1 + 8);
      if ( *(__int64 **)(*a1 + 16) != v8 )
        goto LABEL_7;
      v10 = &v8[*(unsigned int *)(v6 + 28)];
      v11 = *(_DWORD *)(v6 + 28);
      if ( v8 == v10 )
        break;
      v12 = 0;
      while ( v7 != *v8 )
      {
        if ( *v8 == -2 )
        {
          v12 = v8;
          if ( v8 + 1 == v10 )
            goto LABEL_18;
          ++v8;
        }
        else if ( v10 == ++v8 )
        {
          if ( !v12 )
            goto LABEL_21;
LABEL_18:
          *v12 = v7;
          --*(_DWORD *)(v6 + 32);
          ++*(_QWORD *)v6;
          goto LABEL_8;
        }
      }
    }
LABEL_21:
    if ( v11 < *(_DWORD *)(v6 + 24) )
      break;
LABEL_7:
    sub_16CCBA0(v6, v7);
    if ( v9 )
      goto LABEL_8;
  }
  *(_DWORD *)(v6 + 28) = v11 + 1;
  *v10 = v7;
  ++*(_QWORD *)v6;
LABEL_8:
  v13 = v7;
  v14 = 0;
  sub_1838D20(a1 + 1, (__int64)&v13);
}
