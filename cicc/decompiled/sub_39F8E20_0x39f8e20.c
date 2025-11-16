// Function: sub_39F8E20
// Address: 0x39f8e20
//
__int64 __fastcall sub_39F8E20(__int64 a1, char *a2)
{
  char *v2; // rbx
  char *v3; // r14
  int v4; // r15d
  char *v5; // rcx
  __int64 v6; // rdx
  __int64 v7; // rax
  unsigned __int8 v8; // bp
  char *v9; // r13
  int v10; // eax
  unsigned __int8 v11; // al
  unsigned __int16 v12; // ax
  unsigned __int8 v13; // al
  __int64 v15; // [rsp+8h] [rbp-50h]
  unsigned __int64 v16[8]; // [rsp+18h] [rbp-40h] BYREF

  if ( !*(_DWORD *)a2 )
    return 0;
  v2 = a2;
  v3 = 0;
  LOBYTE(v4) = 0;
  v15 = 0;
  v5 = 0;
  while ( 1 )
  {
    v7 = *((int *)v2 + 1);
    if ( (_DWORD)v7 )
      break;
LABEL_9:
    v2 += *(unsigned int *)v2 + 4;
    if ( !*(_DWORD *)v2 )
      return v15;
  }
  v8 = v4;
  v9 = &v2[-v7 + 4];
  if ( v9 == v5 )
  {
LABEL_21:
    sub_39F8BA0(v4, v3, v2 + 8, v16);
    if ( v8 == 0xFF )
    {
      v6 = 0;
    }
    else
    {
      v13 = v8 & 7;
      if ( (v8 & 7) == 2 )
      {
        v6 = 0xFFFF;
      }
      else
      {
        if ( v13 <= 2u )
        {
          if ( v13 )
            goto LABEL_38;
        }
        else
        {
          v6 = 0xFFFFFFFFLL;
          if ( v13 == 3 )
            goto LABEL_5;
          if ( v13 != 4 )
            goto LABEL_38;
        }
        v6 = -1;
      }
    }
LABEL_5:
    v5 = v9;
    if ( (v6 & v16[0]) != 0 )
    {
      ++v15;
      if ( *(_QWORD *)a1 > v16[0] )
        *(_QWORD *)a1 = v16[0];
      v5 = v9;
    }
    goto LABEL_9;
  }
  v10 = sub_39F8CF0((__int64)&v2[-v7 + 4]);
  v4 = v10;
  if ( v10 != 255 )
  {
    v8 = v10;
    if ( (_BYTE)v10 != 0xFF )
    {
      v11 = v10 & 0x70;
      if ( v11 == 32 )
      {
        v3 = *(char **)(a1 + 8);
LABEL_18:
        v12 = *(_WORD *)(a1 + 32);
        if ( (v12 & 0x7F8) == 0x7F8 )
        {
          *(_WORD *)(a1 + 32) = (8 * v8) | v12 & 0xF807;
        }
        else if ( (unsigned __int8)(v12 >> 3) != v4 )
        {
          *(_BYTE *)(a1 + 32) |= 4u;
        }
        goto LABEL_21;
      }
      if ( v11 <= 0x20u )
      {
        if ( (v4 & 0x60) != 0 )
LABEL_38:
          abort();
      }
      else
      {
        if ( v11 == 48 )
        {
          v3 = *(char **)(a1 + 16);
          goto LABEL_18;
        }
        if ( v11 != 80 )
          goto LABEL_38;
      }
    }
    v3 = 0;
    goto LABEL_18;
  }
  return -1;
}
