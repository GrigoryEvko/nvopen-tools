// Function: sub_F7A2A0
// Address: 0xf7a2a0
//
void __fastcall sub_F7A2A0(char *src, char *a2)
{
  char *v2; // r9
  char *v4; // r15
  char *v5; // rbx
  __int64 v6; // rcx
  __int64 v7; // r11
  __int64 v8; // rdi
  char v9; // r14
  char v10; // al
  char *i; // rbx
  __int64 v12; // r12
  __int64 v13; // rdi
  char v14; // al
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rax
  char *v17; // rbx
  char *v18; // [rsp+0h] [rbp-70h]
  char *v19; // [rsp+0h] [rbp-70h]
  __int64 v20; // [rsp+8h] [rbp-68h]
  __int64 v21; // [rsp+8h] [rbp-68h]
  __int64 v22; // [rsp+8h] [rbp-68h]
  __int64 v23; // [rsp+10h] [rbp-60h]
  __int64 v24; // [rsp+10h] [rbp-60h]
  char *v25; // [rsp+10h] [rbp-60h]
  unsigned __int64 v26; // [rsp+30h] [rbp-40h]
  unsigned __int64 v27; // [rsp+30h] [rbp-40h]

  if ( src != a2 )
  {
    v2 = src + 8;
    if ( a2 != src + 8 )
    {
      v4 = src + 16;
      do
      {
        while ( 1 )
        {
          v6 = *(_QWORD *)v2;
          v7 = *(_QWORD *)(*(_QWORD *)v2 + 8LL);
          v8 = *(_QWORD *)(*(_QWORD *)src + 8LL);
          v9 = *(_BYTE *)(v7 + 8);
          v10 = *(_BYTE *)(v8 + 8);
          if ( v9 != 12 )
            break;
          if ( v10 != 12 )
            goto LABEL_8;
          v19 = v2;
          v21 = *(_QWORD *)v2;
          v24 = *(_QWORD *)(*(_QWORD *)v2 + 8LL);
          v27 = sub_BCAE30(v8);
          v16 = sub_BCAE30(v24);
          v7 = v24;
          v6 = v21;
          v2 = v19;
          if ( v27 >= v16 )
            goto LABEL_8;
LABEL_15:
          v17 = v4;
          if ( src != v2 )
          {
            v22 = v6;
            v25 = v2;
            memmove(src + 8, src, v2 - src);
            v6 = v22;
            v2 = v25;
          }
          *(_QWORD *)src = v6;
          v2 += 8;
          v4 += 8;
          if ( a2 == v17 )
            return;
        }
        if ( v10 == 12 )
          goto LABEL_15;
LABEL_8:
        for ( i = v2; ; i -= 8 )
        {
          v12 = *((_QWORD *)i - 1);
          v13 = *(_QWORD *)(v12 + 8);
          v14 = *(_BYTE *)(v13 + 8);
          if ( v9 == 12 )
            break;
          if ( v14 != 12 )
            goto LABEL_5;
LABEL_12:
          *(_QWORD *)i = v12;
          v7 = *(_QWORD *)(v6 + 8);
          v9 = *(_BYTE *)(v7 + 8);
        }
        if ( v14 == 12 )
        {
          v18 = v2;
          v23 = v6;
          v20 = v7;
          v26 = sub_BCAE30(v13);
          v15 = sub_BCAE30(v20);
          v6 = v23;
          v2 = v18;
          if ( v26 < v15 )
            goto LABEL_12;
        }
LABEL_5:
        *(_QWORD *)i = v6;
        v2 += 8;
        v5 = v4;
        v4 += 8;
      }
      while ( a2 != v5 );
    }
  }
}
