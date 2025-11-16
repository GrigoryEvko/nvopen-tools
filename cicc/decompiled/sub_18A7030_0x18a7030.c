// Function: sub_18A7030
// Address: 0x18a7030
//
void __fastcall sub_18A7030(char *src, char *a2)
{
  char *i; // rbx
  __int64 v3; // r15
  _QWORD *v4; // rsi
  unsigned __int64 j; // r14
  __int64 v6; // rax
  __int64 v7; // r13
  unsigned int v8; // ecx
  __int64 v9; // r8
  __int64 v10; // r13
  unsigned __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // r13
  unsigned int v14; // ecx
  __int64 v15; // r8
  __int64 v16; // rax
  __int64 k; // [rsp+10h] [rbp-40h]
  __int64 v19; // [rsp+18h] [rbp-38h]
  __int64 v20; // [rsp+18h] [rbp-38h]

  if ( src != a2 )
  {
    for ( i = src + 8; a2 != i; i += 8 )
    {
      v3 = *(_QWORD *)i;
      v4 = *(_QWORD **)src;
      j = *(_QWORD *)(*(_QWORD *)i + 120LL);
      if ( *(_QWORD *)(*(_QWORD *)i + 72LL) )
      {
        v6 = *(_QWORD *)(v3 + 56);
        if ( !j
          || (v7 = *(_QWORD *)(v3 + 104), v8 = *(_DWORD *)(v7 + 32), *(_DWORD *)(v6 + 32) < v8)
          || *(_DWORD *)(v6 + 32) == v8 && *(_DWORD *)(v6 + 36) < *(_DWORD *)(v7 + 36) )
        {
          j = *(_QWORD *)(v6 + 40);
          goto LABEL_9;
        }
      }
      else
      {
        if ( !j )
          goto LABEL_9;
        v7 = *(_QWORD *)(v3 + 104);
      }
      v9 = *(_QWORD *)(v7 + 64);
      v10 = v7 + 48;
      for ( j = 0; v10 != v9; v9 = sub_220EF30(v19) )
      {
        v19 = v9;
        j += sub_18A58D0(v9 + 64);
      }
LABEL_9:
      v11 = v4[15];
      if ( v4[9] )
      {
        v12 = v4[7];
        if ( v11 )
        {
          v13 = v4[13];
          v14 = *(_DWORD *)(v13 + 32);
          if ( *(_DWORD *)(v12 + 32) >= v14
            && (*(_DWORD *)(v12 + 32) != v14 || *(_DWORD *)(v12 + 36) >= *(_DWORD *)(v13 + 36)) )
          {
            goto LABEL_13;
          }
        }
        v11 = *(_QWORD *)(v12 + 40);
      }
      else if ( v11 )
      {
        v13 = v4[13];
LABEL_13:
        v15 = *(_QWORD *)(v13 + 64);
        v16 = v13 + 48;
        v11 = 0;
        for ( k = v16; k != v15; v15 = sub_220EF30(v20) )
        {
          v20 = v15;
          v11 += sub_18A58D0(v15 + 64);
        }
      }
      if ( v11 >= j )
      {
        sub_18A6D30(i);
      }
      else
      {
        if ( src != i )
          memmove(src + 8, src, i - src);
        *(_QWORD *)src = v3;
      }
    }
  }
}
