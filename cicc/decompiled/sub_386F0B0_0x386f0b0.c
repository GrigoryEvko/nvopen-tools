// Function: sub_386F0B0
// Address: 0x386f0b0
//
void __fastcall sub_386F0B0(__int64 **src, __int64 **a2)
{
  __int64 **i; // r15
  __int64 *v4; // rcx
  __int64 v5; // r8
  __int64 v6; // rdi
  char v7; // r13
  char v8; // al
  __int64 **v9; // rbx
  __int64 *v10; // r12
  char v11; // al
  unsigned int v12; // r13d
  unsigned int v13; // eax
  unsigned int v14; // r12d
  unsigned int v15; // eax
  __int64 **v16; // rbx
  __int64 v17; // [rsp+8h] [rbp-48h]
  __int64 *v18; // [rsp+8h] [rbp-48h]
  __int64 *v19; // [rsp+10h] [rbp-40h]
  __int64 v20; // [rsp+10h] [rbp-40h]
  __int64 *v21; // [rsp+10h] [rbp-40h]

  if ( src != a2 && a2 != src + 1 )
  {
    for ( i = src + 1; ; ++i )
    {
LABEL_6:
      v4 = *i;
      v5 = **i;
      v6 = **src;
      v7 = *(_BYTE *)(v5 + 8);
      v8 = *(_BYTE *)(v6 + 8);
      if ( v7 != 11 )
      {
        v9 = i;
        if ( v8 != 11 )
          goto LABEL_8;
        goto LABEL_14;
      }
      v9 = i;
      if ( v8 != 11 )
        break;
      v18 = *i;
      v20 = **i;
      v14 = sub_1643030(v6);
      v15 = sub_1643030(v20);
      v5 = v20;
      v4 = v18;
      if ( v14 >= v15 )
        break;
LABEL_14:
      v16 = i + 1;
      if ( src != i )
      {
        v21 = v4;
        memmove(src + 1, src, (char *)i - (char *)src);
        v4 = v21;
      }
      *src = v4;
      if ( a2 == v16 )
        return;
    }
    while ( 1 )
    {
LABEL_8:
      v10 = *(v9 - 1);
      v11 = *(_BYTE *)(*v10 + 8);
      if ( v7 == 11 )
      {
        if ( v11 != 11 || (v19 = v4, v17 = v5, v12 = sub_1643030(*v10), v13 = sub_1643030(v17), v4 = v19, v12 >= v13) )
        {
LABEL_5:
          *v9 = v4;
          if ( a2 == ++i )
            return;
          goto LABEL_6;
        }
      }
      else if ( v11 != 11 )
      {
        goto LABEL_5;
      }
      *v9 = v10;
      v5 = *v4;
      --v9;
      v7 = *(_BYTE *)(*v4 + 8);
    }
  }
}
