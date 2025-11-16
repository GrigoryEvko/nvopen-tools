// Function: sub_8E59D0
// Address: 0x8e59d0
//
unsigned __int8 *__fastcall sub_8E59D0(unsigned __int8 *a1, __int64 a2)
{
  unsigned __int8 *v2; // r15
  unsigned __int8 v4; // al
  unsigned __int64 v5; // r12
  unsigned __int64 v6; // rsi
  unsigned __int8 v7; // al
  unsigned __int64 v8; // r13
  char *v9; // r12
  int v10; // ebx
  int v11; // ebx
  double v12; // xmm0_8
  char *v14; // rdx
  unsigned __int64 v15; // [rsp+8h] [rbp-88h]
  double v16; // [rsp+18h] [rbp-78h] BYREF
  char s[112]; // [rsp+20h] [rbp-70h] BYREF

  v2 = a1;
  v4 = *a1;
  v16 = 0.0;
  if ( v4 != 69 && v4 != 95 && v4 )
  {
    v5 = 0;
    do
    {
      v6 = v5++;
      v7 = a1[v5];
    }
    while ( v7 != 95 && v7 != 69 && v7 );
    if ( (v5 & 1) != 0 )
    {
      if ( !*(_DWORD *)(a2 + 24) )
      {
        ++*(_QWORD *)(a2 + 32);
        ++*(_QWORD *)(a2 + 48);
        *(_DWORD *)(a2 + 24) = 1;
      }
      v5 = v6;
      if ( v6 > 0x11 )
        goto LABEL_8;
    }
    else if ( v5 > 0x11 )
    {
LABEL_8:
      v15 = 8;
      if ( !*(_DWORD *)(a2 + 24) )
      {
        ++*(_QWORD *)(a2 + 32);
        ++*(_QWORD *)(a2 + 48);
        *(_DWORD *)(a2 + 24) = 1;
      }
LABEL_10:
      v8 = 0;
      v9 = &s[v15 - 8];
      while ( 1 )
      {
        v10 = sub_8E58A0(*v2, a2);
        if ( *(_DWORD *)(a2 + 24) )
          return v2;
        v11 = sub_8E58A0(v2[1], a2) | (16 * v10);
        if ( *(_DWORD *)(a2 + 24) )
          return v2;
        if ( unk_4F07580 )
          *(v9 - 1) = v11;
        else
          s[v8 - 8] = v11;
        ++v8;
        v2 += 2;
        --v9;
        if ( v8 >= v15 )
        {
          if ( v8 <= 4 )
          {
            v12 = *(float *)&v16;
            goto LABEL_25;
          }
          sprintf(s, "%.*g", 15, v16);
          goto LABEL_26;
        }
      }
    }
    v15 = v5 >> 1;
    if ( v5 >> 1 )
      goto LABEL_10;
  }
  if ( !*(_DWORD *)(a2 + 24) )
  {
    v12 = 0.0;
LABEL_25:
    sprintf(s, "%.*g", 6, v12);
LABEL_26:
    if ( !strchr(s, 46) && !strchr(s, 101) )
    {
      v14 = &s[strlen(s) - 1];
      if ( (unsigned int)(unsigned __int8)*v14 - 48 <= 9 )
        strcpy(v14 + 1, ".0");
    }
    if ( !*(_QWORD *)(a2 + 32) )
      sub_8E5790((unsigned __int8 *)s, a2);
  }
  return v2;
}
