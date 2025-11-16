// Function: sub_39F90C0
// Address: 0x39f90c0
//
__int64 __fastcall sub_39F90C0(__int64 a1, __int64 a2, __int64 a3)
{
  char v4; // al
  char *v5; // rdx
  unsigned __int8 v6; // cl
  char *v7; // rsi
  char v8; // al
  unsigned __int8 v9; // cl
  char *v10; // rsi
  __int64 result; // rax
  unsigned __int64 v12; // [rsp+0h] [rbp-28h] BYREF
  unsigned __int64 v13[4]; // [rsp+8h] [rbp-20h] BYREF

  v4 = sub_39F8CF0(a2 + 4 - *(int *)(a2 + 4));
  v5 = (char *)(a2 + 8);
  if ( v4 == -1 )
  {
LABEL_17:
    v7 = 0;
    goto LABEL_6;
  }
  v6 = v4 & 0x70;
  if ( (v4 & 0x70) != 0x20 )
  {
    if ( v6 <= 0x20u )
    {
      if ( (v4 & 0x60) != 0 )
        goto LABEL_24;
    }
    else
    {
      if ( v6 == 48 )
      {
        v7 = *(char **)(a1 + 16);
        goto LABEL_6;
      }
      if ( v6 != 80 )
        goto LABEL_24;
    }
    goto LABEL_17;
  }
  v7 = *(char **)(a1 + 8);
LABEL_6:
  sub_39F8BA0(v4, v7, v5, &v12);
  v8 = sub_39F8CF0(a3 + 4 - *(int *)(a3 + 4));
  if ( v8 == -1 )
    goto LABEL_15;
  v9 = v8 & 0x70;
  if ( (v8 & 0x70) != 0x20 )
  {
    if ( v9 > 0x20u )
    {
      if ( v9 == 48 )
      {
        v10 = *(char **)(a1 + 16);
        goto LABEL_11;
      }
      if ( v9 == 80 )
        goto LABEL_15;
LABEL_24:
      abort();
    }
    if ( (v8 & 0x60) != 0 )
      goto LABEL_24;
LABEL_15:
    v10 = 0;
    goto LABEL_11;
  }
  v10 = *(char **)(a1 + 8);
LABEL_11:
  sub_39F8BA0(v8, v10, (char *)(a3 + 8), v13);
  result = 1;
  if ( v12 <= v13[0] )
    return (unsigned int)-(v12 < v13[0]);
  return result;
}
