// Function: sub_386F200
// Address: 0x386f200
//
__int64 __fastcall sub_386F200(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 result; // rax
  __int64 v5; // r11
  __int64 i; // r15
  __int64 v9; // rbx
  __int64 **v10; // r13
  __int64 *v11; // rdx
  __int64 v12; // rdi
  char v13; // al
  unsigned int v14; // eax
  __int64 v15; // r15
  __int64 *v16; // r14
  __int64 v17; // rbx
  __int64 *v18; // rax
  __int64 v19; // [rsp+8h] [rbp-68h]
  __int64 *v20; // [rsp+10h] [rbp-60h]
  __int64 v21; // [rsp+20h] [rbp-50h]
  __int64 *v22; // [rsp+20h] [rbp-50h]
  __int64 *v23; // [rsp+28h] [rbp-48h]
  unsigned int v24; // [rsp+30h] [rbp-40h]
  __int64 v25; // [rsp+30h] [rbp-40h]
  __int64 v26; // [rsp+38h] [rbp-38h]
  unsigned int v27; // [rsp+38h] [rbp-38h]

  result = a3 & 1;
  v5 = (a3 - 1) / 2;
  v26 = result;
  if ( a2 < v5 )
  {
    for ( i = a2; ; i = v9 )
    {
      v9 = 2 * (i + 1);
      v10 = (__int64 **)(a1 + 16 * (i + 1));
      v11 = *v10;
      v12 = **(v10 - 1);
      v13 = *(_BYTE *)(v12 + 8);
      if ( *(_BYTE *)(**v10 + 8) == 11 )
      {
        if ( v13 != 11 )
          goto LABEL_4;
        v19 = v5;
        v20 = a4;
        v23 = *v10;
        v21 = **v10;
        v24 = sub_1643030(v12);
        v14 = sub_1643030(v21);
        v11 = v23;
        a4 = v20;
        v5 = v19;
        if ( v24 >= v14 )
        {
LABEL_4:
          *(_QWORD *)(a1 + 8 * i) = v11;
          if ( v9 >= v5 )
            goto LABEL_10;
          continue;
        }
      }
      else if ( v13 != 11 )
      {
        goto LABEL_4;
      }
      --v9;
      v10 = (__int64 **)(a1 + 8 * v9);
      *(_QWORD *)(a1 + 8 * i) = *v10;
      if ( v9 >= v5 )
      {
LABEL_10:
        if ( v26 )
          goto LABEL_11;
        goto LABEL_18;
      }
    }
  }
  v10 = (__int64 **)(a1 + 8 * a2);
  if ( (a3 & 1) != 0 )
    goto LABEL_15;
  v9 = a2;
LABEL_18:
  if ( (a3 - 2) / 2 == v9 )
  {
    v17 = 2 * v9 + 2;
    v18 = *(__int64 **)(a1 + 8 * v17 - 8);
    v9 = v17 - 1;
    *v10 = v18;
    v10 = (__int64 **)(a1 + 8 * v9);
  }
LABEL_11:
  result = v9 - 1;
  v15 = (v9 - 1) / 2;
  if ( v9 > a2 )
  {
    while ( 1 )
    {
      v10 = (__int64 **)(a1 + 8 * v15);
      v16 = *v10;
      result = *(unsigned __int8 *)(*a4 + 8);
      if ( *(_BYTE *)(**v10 + 8) == 11 )
      {
        if ( (_BYTE)result != 11
          || (v22 = a4,
              v25 = **v10,
              v27 = sub_1643030(*a4),
              result = sub_1643030(v25),
              a4 = v22,
              v27 >= (unsigned int)result) )
        {
LABEL_14:
          v10 = (__int64 **)(a1 + 8 * v9);
          break;
        }
      }
      else if ( (_BYTE)result != 11 )
      {
        goto LABEL_14;
      }
      *(_QWORD *)(a1 + 8 * v9) = v16;
      v9 = v15;
      result = (v15 - 1) / 2;
      if ( a2 >= v15 )
        break;
      v15 = (v15 - 1) / 2;
    }
  }
LABEL_15:
  *v10 = a4;
  return result;
}
