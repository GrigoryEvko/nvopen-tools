// Function: sub_39F9490
// Address: 0x39f9490
//
unsigned int *__fastcall sub_39F9490(__int64 a1, unsigned int *a2, __int64 a3)
{
  unsigned int *v4; // rbx
  __int16 v5; // ax
  int v6; // ebp
  unsigned __int8 v7; // dl
  char *v8; // r13
  unsigned int *v9; // r15
  int v10; // eax
  char v11; // r10
  unsigned __int8 v12; // al
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rdx
  __int64 v15; // rax
  char *v16; // rax
  char v17; // r10
  unsigned __int8 v18; // al
  __int64 v19; // rdx
  char *v21; // rax
  char *v22; // rax
  __int64 v23; // [rsp+8h] [rbp-50h]
  char v24; // [rsp+8h] [rbp-50h]
  unsigned __int64 v25; // [rsp+10h] [rbp-48h] BYREF
  unsigned __int64 v26[8]; // [rsp+18h] [rbp-40h] BYREF

  v4 = a2;
  v5 = *(_WORD *)(a1 + 32) >> 3;
  if ( (_BYTE)v5 == 0xFF )
  {
    v8 = 0;
    v6 = 255;
  }
  else
  {
    v6 = (unsigned __int8)v5;
    v7 = v5 & 0x70;
    if ( (v5 & 0x70) != 0x20 )
    {
      if ( v7 <= 0x20u )
      {
        if ( (v5 & 0x60) == 0 )
        {
LABEL_37:
          v8 = 0;
          goto LABEL_6;
        }
      }
      else
      {
        if ( v7 == 48 )
        {
          v8 = *(char **)(a1 + 16);
          goto LABEL_6;
        }
        if ( v7 == 80 )
          goto LABEL_37;
      }
LABEL_48:
      abort();
    }
    v8 = *(char **)(a1 + 8);
  }
LABEL_6:
  v9 = 0;
  if ( *a2 )
  {
    do
    {
      v15 = (int)v4[1];
      if ( !(_DWORD)v15 )
        goto LABEL_16;
      if ( (*(_BYTE *)(a1 + 32) & 4) == 0 || (unsigned int *)((char *)v4 - v15 + 4) == v9 )
      {
        if ( v6 )
          goto LABEL_20;
      }
      else
      {
        v23 = (__int64)v4 - v15 + 4;
        v10 = sub_39F8CF0(v23);
        v6 = v10;
        v11 = v10;
        if ( (_BYTE)v10 == 0xFF )
        {
          v8 = 0;
          v22 = sub_39F8BA0(255, 0, (char *)v4 + 8, &v25);
          sub_39F8BA0(v6 & 0xF, 0, v22, v26);
          v19 = 0;
          v9 = (unsigned int *)v23;
          goto LABEL_24;
        }
        v12 = v10 & 0x70;
        if ( v12 == 32 )
        {
          v8 = *(char **)(a1 + 8);
          v9 = (unsigned int *)v23;
          if ( v6 )
          {
LABEL_43:
            v24 = v11;
            v21 = sub_39F8BA0(v6, v8, (char *)v4 + 8, &v25);
            sub_39F8BA0(v6 & 0xF, 0, v21, v26);
            v17 = v24;
LABEL_21:
            v18 = v17 & 7;
            if ( (v17 & 7) == 2 )
            {
              v19 = 0xFFFF;
              goto LABEL_24;
            }
            if ( v18 <= 2u )
            {
              if ( v18 )
                goto LABEL_48;
            }
            else
            {
              v19 = 0xFFFFFFFFLL;
              if ( v18 == 3 )
                goto LABEL_24;
              if ( v18 != 4 )
                goto LABEL_48;
            }
            v19 = -1;
            goto LABEL_24;
          }
        }
        else
        {
          if ( v12 <= 0x20u )
          {
            if ( (v11 & 0x60) != 0 )
              goto LABEL_48;
          }
          else
          {
            if ( v12 == 48 )
            {
              v8 = *(char **)(a1 + 16);
              v9 = (unsigned int *)v23;
              if ( v6 )
                goto LABEL_43;
              goto LABEL_14;
            }
            if ( v12 != 80 )
              goto LABEL_48;
          }
          v9 = (unsigned int *)v23;
          v8 = 0;
          if ( v6 )
          {
LABEL_20:
            v16 = sub_39F8BA0(v6, v8, (char *)v4 + 8, &v25);
            sub_39F8BA0(v6 & 0xF, 0, v16, v26);
            v17 = v6;
            if ( (_BYTE)v6 != 0xFF )
              goto LABEL_21;
            v19 = 0;
LABEL_24:
            if ( (v19 & v25) != 0 && a3 - v25 < v26[0] )
              return v4;
            goto LABEL_16;
          }
        }
      }
LABEL_14:
      v13 = *((_QWORD *)v4 + 1);
      v14 = *((_QWORD *)v4 + 2);
      v6 = 0;
      v25 = v13;
      v26[0] = v14;
      if ( v13 && a3 - v13 < v14 )
        return v4;
LABEL_16:
      v4 = (unsigned int *)((char *)v4 + *v4 + 4);
    }
    while ( *v4 );
  }
  return 0;
}
