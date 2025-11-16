// Function: sub_694AA0
// Address: 0x694aa0
//
__int64 __fastcall sub_694AA0(int a1, int a2, int a3, int a4, __int64 a5)
{
  int v5; // r10d
  int v6; // r11d
  __int64 *v9; // r13
  bool v10; // bl
  _BOOL4 v11; // r15d
  __int64 v12; // rdx
  char v13; // al
  int v14; // r10d
  int v15; // r11d
  char v16; // cl
  int v17; // r9d
  int v18; // eax
  int v19; // r9d
  int v20; // eax
  char *v21; // rdx
  char v22; // bl
  __int64 result; // rax
  int v24; // eax
  __int64 v25; // rax
  int v26; // eax
  int v27; // [rsp+8h] [rbp-178h]
  int v28; // [rsp+10h] [rbp-170h]
  __int64 v29; // [rsp+18h] [rbp-168h]
  int v30; // [rsp+20h] [rbp-160h]
  __int64 v32; // [rsp+28h] [rbp-158h]
  __int64 v33; // [rsp+28h] [rbp-158h]
  __int64 v34; // [rsp+38h] [rbp-148h] BYREF
  char v35[8]; // [rsp+40h] [rbp-140h] BYREF
  int v36; // [rsp+48h] [rbp-138h]
  _BYTE v37[208]; // [rsp+B0h] [rbp-D0h] BYREF

  v5 = a1;
  v6 = a2;
  v9 = *(__int64 **)(a5 + 16);
  v10 = (*(_BYTE *)(a5 + 42) & 2) != 0;
  v11 = v10;
  if ( a3 )
  {
    v12 = *v9;
    v13 = *(_BYTE *)(*v9 + 80);
    if ( v13 == 9 || v13 == 7 )
    {
      v29 = *(_QWORD *)(v12 + 88);
    }
    else
    {
      if ( v13 != 21 )
        BUG();
      v29 = *(_QWORD *)(*(_QWORD *)(v12 + 88) + 192LL);
    }
    if ( (*(_BYTE *)(v29 + 89) & 1) == 0 || (v26 = sub_6EA1E0(v29), v5 = a1, v6 = a2, v26) )
      v24 = 142606336;
    else
      v24 = 0x8000000;
    v27 = v6;
    v28 = v5;
    v30 = v24;
    sub_6E2250(v37, &v34, 4, v10, v9, a5);
    v18 = v30;
    v14 = v28;
    v15 = v27;
  }
  else
  {
    sub_6E2250(v37, &v34, 4, v10, v9, a5);
    v14 = a1;
    v15 = a2;
    if ( !v9 )
    {
      v29 = 0;
      v16 = *(_BYTE *)(a5 + 40);
      if ( (*(_BYTE *)(a5 + 43) & 0x10) != 0 )
        v17 = 136314880;
      else
        v17 = 0x8000000;
      goto LABEL_14;
    }
    v29 = 0;
    v18 = 0x8000000;
  }
  if ( *v9 && *(_BYTE *)(*v9 + 80) == 8 )
  {
    BYTE1(v18) |= 0x10u;
    v16 = *(_BYTE *)(a5 + 40);
    v17 = v18;
  }
  else
  {
    v16 = *(_BYTE *)(a5 + 40);
    v19 = v18;
    v20 = v18 | 0x201;
    v17 = v19 | 1;
    if ( (v16 & 2) != 0 )
      v17 = v20;
  }
LABEL_14:
  v21 = *(char **)(a5 + 32);
  if ( v21 )
    v21 = v35;
  sub_839D30(v14, v15, v16 & 1, 0, 0, v17, a4, 0, 0, 0, a5, (__int64)v21);
  if ( *(_QWORD *)(a5 + 32) )
  {
    if ( v36 == 7 )
      *(_BYTE *)(a5 + 41) |= 2u;
    else
      sub_832CF0(v35);
  }
  v22 = ((*(_BYTE *)(a5 + 40) >> 6) ^ 1) & 1 & v10;
  if ( (*(_BYTE *)(a5 + 41) & 2) != 0 )
  {
    if ( v22 )
      sub_6E2A90();
    return sub_6E2C70(v34, v11, v9, a5);
  }
  v25 = *(_QWORD *)(a5 + 8);
  if ( v22 )
  {
    if ( !v25 )
    {
      if ( *(_QWORD *)a5 )
        sub_6E2AC0(*(_QWORD *)a5);
      return sub_6E2C70(v34, v11, v9, a5);
    }
    v33 = *(_QWORD *)(a5 + 8);
    sub_6E2920(v33);
    v25 = v33;
  }
  v32 = v25;
  sub_6E2C70(v34, v11, v9, a5);
  result = v32;
  if ( a3 != 0 && v32 != 0 && (*(_BYTE *)(a5 + 40) & 0x40) == 0 )
    *(_QWORD *)(v32 + 8) = v29;
  return result;
}
