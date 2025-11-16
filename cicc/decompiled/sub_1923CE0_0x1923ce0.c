// Function: sub_1923CE0
// Address: 0x1923ce0
//
void __fastcall sub_1923CE0(__int64 a1)
{
  __int64 v2; // r13
  __int64 i; // r14
  __int64 v4; // rbx
  __int64 v5; // rbx
  __int64 *v6; // rax
  char v7; // dl
  __int64 *v8; // rcx
  unsigned int v9; // edi
  __int64 *v10; // rsi
  __int64 v11; // [rsp+0h] [rbp-40h] BYREF
  char v12; // [rsp+10h] [rbp-30h]

  v2 = *(_QWORD *)(a1 + 112);
  if ( !*(_BYTE *)(v2 - 8) )
    goto LABEL_25;
  while ( 1 )
  {
    while ( 1 )
    {
LABEL_2:
      for ( i = *(_QWORD *)(v2 - 16); !i; *(_QWORD *)(v2 - 16) = i )
      {
        *(_QWORD *)(a1 + 112) -= 24LL;
        v2 = *(_QWORD *)(a1 + 112);
        if ( v2 == *(_QWORD *)(a1 + 104) )
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
      v5 = sub_1648700(i)[5];
      v6 = *(__int64 **)(a1 + 8);
      if ( *(__int64 **)(a1 + 16) != v6 )
        goto LABEL_7;
      v8 = &v6[*(unsigned int *)(a1 + 28)];
      v9 = *(_DWORD *)(a1 + 28);
      if ( v6 == v8 )
        break;
      v10 = 0;
      while ( v5 != *v6 )
      {
        if ( *v6 == -2 )
        {
          v10 = v6;
          if ( v6 + 1 == v8 )
            goto LABEL_18;
          ++v6;
        }
        else if ( v8 == ++v6 )
        {
          if ( !v10 )
            goto LABEL_21;
LABEL_18:
          *v10 = v5;
          --*(_DWORD *)(a1 + 32);
          ++*(_QWORD *)a1;
          goto LABEL_8;
        }
      }
    }
LABEL_21:
    if ( v9 < *(_DWORD *)(a1 + 24) )
      break;
LABEL_7:
    sub_16CCBA0(a1, v5);
    if ( v7 )
      goto LABEL_8;
  }
  *(_DWORD *)(a1 + 28) = v9 + 1;
  *v8 = v5;
  ++*(_QWORD *)a1;
LABEL_8:
  v11 = v5;
  v12 = 0;
  sub_1923AF0((unsigned __int64 *)(a1 + 104), (__int64)&v11);
}
