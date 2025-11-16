// Function: sub_39F91E0
// Address: 0x39f91e0
//
__int16 __fastcall sub_39F91E0(__int64 a1, __int64 *a2, char *a3)
{
  int v4; // eax
  int v5; // ebp
  unsigned __int8 v6; // dl
  char *v7; // r13
  unsigned int *v8; // r15
  __int64 v9; // rax
  int v10; // eax
  char v11; // r10
  unsigned __int8 v12; // al
  __int64 v13; // rax
  __int64 v14; // rdx
  char v15; // r10
  char v16; // dl
  bool v17; // cc
  __int64 v18; // rax
  __int64 v20; // [rsp+8h] [rbp-50h]
  char v21; // [rsp+8h] [rbp-50h]
  unsigned __int64 v22[8]; // [rsp+18h] [rbp-40h] BYREF

  LOWORD(v4) = *(_WORD *)(a1 + 32) >> 3;
  if ( (_BYTE)v4 == 0xFF )
  {
    v7 = 0;
    v5 = 255;
  }
  else
  {
    v5 = (unsigned __int8)v4;
    v6 = v4 & 0x70;
    if ( (v4 & 0x70) != 0x20 )
    {
      if ( v6 <= 0x20u )
      {
        if ( (v4 & 0x60) == 0 )
        {
LABEL_38:
          v7 = 0;
          goto LABEL_6;
        }
      }
      else
      {
        if ( v6 == 48 )
        {
          v7 = *(char **)(a1 + 16);
          goto LABEL_6;
        }
        if ( v6 == 80 )
          goto LABEL_38;
      }
LABEL_47:
      abort();
    }
    v7 = *(char **)(a1 + 8);
  }
LABEL_6:
  v8 = 0;
  while ( *(_DWORD *)a3 )
  {
    v9 = *((int *)a3 + 1);
    if ( !(_DWORD)v9 )
      goto LABEL_18;
    if ( (*(_BYTE *)(a1 + 32) & 4) == 0 || &a3[-v9 + 4] == (char *)v8 )
    {
      if ( !v5 )
        goto LABEL_15;
      sub_39F8BA0(v5, v7, a3 + 8, v22);
      v15 = v5;
      if ( (_BYTE)v5 == 0xFF )
      {
        v18 = 0;
        goto LABEL_25;
      }
    }
    else
    {
      v20 = (__int64)&a3[-v9 + 4];
      v10 = sub_39F8CF0(v20);
      v5 = v10;
      v11 = v10;
      if ( (_BYTE)v10 == 0xFF )
      {
        v7 = 0;
        sub_39F8BA0(255, 0, a3 + 8, v22);
        v18 = 0;
        v8 = (unsigned int *)v20;
        goto LABEL_25;
      }
      v12 = v10 & 0x70;
      if ( v12 == 32 )
      {
        v7 = *(char **)(a1 + 8);
        v8 = (unsigned int *)v20;
        if ( !v5 )
        {
LABEL_15:
          v5 = 0;
          if ( *((_QWORD *)a3 + 1) )
            goto LABEL_16;
          goto LABEL_18;
        }
      }
      else
      {
        if ( v12 <= 0x20u )
        {
          if ( (v11 & 0x60) != 0 )
            goto LABEL_47;
          goto LABEL_32;
        }
        if ( v12 != 48 )
        {
          if ( v12 != 80 )
            goto LABEL_47;
LABEL_32:
          v8 = (unsigned int *)v20;
          v7 = 0;
          if ( !v5 )
            goto LABEL_15;
          sub_39F8BA0(v5, 0, a3 + 8, v22);
          v16 = v5 & 7;
          v17 = (v5 & 7u) <= 2;
          if ( (v5 & 7) == 2 )
          {
LABEL_34:
            v18 = 0xFFFF;
            goto LABEL_25;
          }
          goto LABEL_23;
        }
        v7 = *(char **)(a1 + 16);
        v8 = (unsigned int *)v20;
        if ( !v5 )
          goto LABEL_15;
      }
      v21 = v11;
      sub_39F8BA0(v5, v7, a3 + 8, v22);
      v15 = v21;
    }
    v16 = v15 & 7;
    v17 = (v15 & 7u) <= 2;
    if ( (v15 & 7) == 2 )
      goto LABEL_34;
LABEL_23:
    if ( v17 )
    {
      if ( v16 )
        goto LABEL_47;
    }
    else
    {
      v18 = 0xFFFFFFFFLL;
      if ( v16 == 3 )
        goto LABEL_25;
      if ( v16 != 4 )
        goto LABEL_47;
    }
    v18 = -1;
LABEL_25:
    if ( (v22[0] & v18) != 0 )
    {
LABEL_16:
      v13 = *a2;
      if ( *a2 )
      {
        v14 = *(_QWORD *)(v13 + 8);
        *(_QWORD *)(v13 + 8) = v14 + 1;
        *(_QWORD *)(v13 + 8 * v14 + 16) = a3;
      }
    }
LABEL_18:
    a3 += *(unsigned int *)a3 + 4;
    v4 = *(_DWORD *)a3;
  }
  return v4;
}
