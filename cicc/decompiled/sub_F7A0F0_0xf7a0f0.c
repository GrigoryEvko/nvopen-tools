// Function: sub_F7A0F0
// Address: 0xf7a0f0
//
char *__fastcall sub_F7A0F0(char *src, char *a2, char *a3, char *a4, _QWORD *a5)
{
  char *v7; // r12
  __int64 v9; // rcx
  __int64 v10; // r14
  __int64 v11; // rdi
  char v12; // al
  unsigned __int64 v13; // rax
  char *v14; // rbx
  __int64 v16; // [rsp+8h] [rbp-68h]
  __int64 v17; // [rsp+10h] [rbp-60h]
  unsigned __int64 v18; // [rsp+30h] [rbp-40h]

  v7 = src;
  if ( src != a2 )
  {
    while ( a3 != a4 )
    {
      v9 = *(_QWORD *)a3;
      v10 = *(_QWORD *)v7;
      v11 = *(_QWORD *)(*(_QWORD *)v7 + 8LL);
      v12 = *(_BYTE *)(v11 + 8);
      if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)a3 + 8LL) + 8LL) == 12 )
      {
        if ( v12 == 12 )
        {
          v17 = *(_QWORD *)a3;
          v16 = *(_QWORD *)(*(_QWORD *)a3 + 8LL);
          v18 = sub_BCAE30(v11);
          v13 = sub_BCAE30(v16);
          v9 = v17;
          if ( v18 < v13 )
            goto LABEL_9;
        }
LABEL_4:
        v7 += 8;
        *a5++ = v10;
        if ( v7 == a2 )
          break;
      }
      else
      {
        if ( v12 != 12 )
          goto LABEL_4;
LABEL_9:
        a3 += 8;
        *a5++ = v9;
        if ( v7 == a2 )
          break;
      }
    }
  }
  if ( a2 != v7 )
    memmove(a5, v7, a2 - v7);
  v14 = (char *)a5 + a2 - v7;
  if ( a4 != a3 )
    memmove(v14, a3, a4 - a3);
  return &v14[a4 - a3];
}
