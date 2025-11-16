// Function: sub_25F7210
// Address: 0x25f7210
//
char *__fastcall sub_25F7210(char *src, char *a2, char *a3, char *a4, _QWORD *a5)
{
  char *v5; // r14
  char *v6; // r13
  __int64 v8; // rbx
  __int64 v9; // r15
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // rax
  char *v12; // r12
  unsigned int v14; // ecx
  int v17; // [rsp+1Ch] [rbp-34h]
  int v18; // [rsp+1Ch] [rbp-34h]

  v5 = a3;
  v6 = src;
  if ( src != a2 && a3 != a4 )
  {
    do
    {
      v8 = *(_QWORD *)v5;
      v9 = *(_QWORD *)v6;
      if ( *(_DWORD *)(*(_QWORD *)v5 + 32LL) > 0x40u )
      {
        v10 = -1;
        v17 = *(_DWORD *)(*(_QWORD *)v5 + 32LL);
        if ( v17 - (unsigned int)sub_C444A0(v8 + 24) <= 0x40 )
          v10 = **(_QWORD **)(v8 + 24);
      }
      else
      {
        v10 = *(_QWORD *)(v8 + 24);
      }
      if ( *(_DWORD *)(v9 + 32) > 0x40u )
      {
        v18 = *(_DWORD *)(v9 + 32);
        v14 = v18 - sub_C444A0(v9 + 24);
        v11 = -1;
        if ( v14 <= 0x40 )
          v11 = **(_QWORD **)(v9 + 24);
      }
      else
      {
        v11 = *(_QWORD *)(v9 + 24);
      }
      if ( v11 > v10 )
      {
        *a5 = v8;
        v5 += 8;
        ++a5;
        if ( v6 == a2 )
          break;
      }
      else
      {
        v6 += 8;
        *a5++ = v9;
        if ( v6 == a2 )
          break;
      }
    }
    while ( v5 != a4 );
  }
  if ( a2 != v6 )
    memmove(a5, v6, a2 - v6);
  v12 = (char *)a5 + a2 - v6;
  if ( a4 != v5 )
    memmove(v12, v5, a4 - v5);
  return &v12[a4 - v5];
}
