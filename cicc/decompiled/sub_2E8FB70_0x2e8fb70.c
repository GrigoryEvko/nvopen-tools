// Function: sub_2E8FB70
// Address: 0x2e8fb70
//
__int64 __fastcall sub_2E8FB70(__int64 a1, unsigned int *a2, __int64 a3, __int64 a4)
{
  unsigned __int8 *v4; // rbx
  __int64 result; // rax
  unsigned int *v7; // r13
  unsigned int v8; // r15d
  unsigned int *v9; // r14
  unsigned int v10; // esi
  unsigned int *v11; // r8
  unsigned int v12; // esi
  unsigned int v13; // esi
  unsigned int v14; // esi
  unsigned int *v15; // rbx
  unsigned int v16; // esi
  unsigned int v17; // esi
  unsigned int v18; // esi
  unsigned int v19; // esi
  __int64 v20; // [rsp+8h] [rbp-78h]
  __int64 v21; // [rsp+20h] [rbp-60h]
  unsigned int *v22; // [rsp+28h] [rbp-58h]
  char v24; // [rsp+3Fh] [rbp-41h]
  unsigned __int8 *v25; // [rsp+40h] [rbp-40h]

  v4 = *(unsigned __int8 **)(a1 + 32);
  result = (__int64)&v4[40 * (*(_DWORD *)(a1 + 40) & 0xFFFFFF)];
  v25 = (unsigned __int8 *)result;
  if ( (unsigned __int8 *)result == v4 )
    return result;
  v24 = 0;
  v22 = &a2[a3];
  v20 = (4 * a3) >> 2;
  v7 = (unsigned int *)((char *)a2 + ((4 * a3) & 0xFFFFFFFFFFFFFFF0LL));
  v21 = (4 * a3) >> 4;
  do
  {
    result = *v4;
    if ( (_BYTE)result == 12 )
    {
      v24 = 1;
    }
    else if ( !(_BYTE)result && (v4[3] & 0x10) != 0 )
    {
      v8 = *((_DWORD *)v4 + 2);
      result = v8 - 1;
      if ( (unsigned int)result <= 0x3FFFFFFE )
      {
        if ( v21 > 0 )
        {
          v9 = a2;
          while ( 1 )
          {
            v14 = *v9;
            if ( v8 == *v9 )
              goto LABEL_22;
            result = v14 - 1;
            if ( (unsigned int)result <= 0x3FFFFFFE )
            {
              result = sub_E92070(a4, v14, v8);
              if ( (_BYTE)result )
                goto LABEL_22;
            }
            v10 = v9[1];
            v11 = v9 + 1;
            if ( v8 == v10 )
              goto LABEL_29;
            result = v10 - 1;
            if ( (unsigned int)result <= 0x3FFFFFFE )
            {
              result = sub_E92070(a4, v10, v8);
              v11 = v9 + 1;
              if ( (_BYTE)result )
                goto LABEL_29;
            }
            v12 = v9[2];
            v11 = v9 + 2;
            if ( v8 == v12 )
              goto LABEL_29;
            result = v12 - 1;
            if ( (unsigned int)result <= 0x3FFFFFFE )
            {
              result = sub_E92070(a4, v12, v8);
              v11 = v9 + 2;
              if ( (_BYTE)result )
                goto LABEL_29;
            }
            v13 = v9[3];
            v11 = v9 + 3;
            if ( v8 == v13
              || (result = v13 - 1, (unsigned int)result <= 0x3FFFFFFE)
              && (result = sub_E92070(a4, v13, v8), v11 = v9 + 3, (_BYTE)result) )
            {
LABEL_29:
              if ( v22 != v11 )
                goto LABEL_23;
              goto LABEL_30;
            }
            v9 += 4;
            if ( v9 == v7 )
            {
              result = v22 - v7;
              goto LABEL_33;
            }
          }
        }
        result = v20;
        v9 = a2;
LABEL_33:
        if ( result != 2 )
        {
          if ( result != 3 )
          {
            if ( result != 1 )
            {
LABEL_30:
              v4[3] |= 0x40u;
              goto LABEL_23;
            }
LABEL_36:
            v17 = *v9;
            if ( v8 != *v9 )
            {
              result = v17 - 1;
              if ( (unsigned int)result > 0x3FFFFFFE )
                goto LABEL_30;
              result = sub_E92070(a4, v17, v8);
              if ( !(_BYTE)result )
                goto LABEL_30;
            }
LABEL_22:
            if ( v22 != v9 )
              goto LABEL_23;
            goto LABEL_30;
          }
          v18 = *v9;
          if ( v8 == *v9 )
            goto LABEL_22;
          result = v18 - 1;
          if ( (unsigned int)result <= 0x3FFFFFFE )
          {
            result = sub_E92070(a4, v18, v8);
            if ( (_BYTE)result )
              goto LABEL_22;
          }
          ++v9;
        }
        v19 = *v9;
        if ( v8 != *v9 )
        {
          result = v19 - 1;
          if ( (unsigned int)result > 0x3FFFFFFE || (result = sub_E92070(a4, v19, v8), !(_BYTE)result) )
          {
            ++v9;
            goto LABEL_36;
          }
        }
        goto LABEL_22;
      }
    }
LABEL_23:
    v4 += 40;
  }
  while ( v25 != v4 );
  if ( v24 )
  {
    result = (__int64)a2;
    if ( a2 != v22 )
    {
      v15 = a2;
      do
      {
        v16 = *v15++;
        result = sub_2E8FA40(a1, v16, a4);
      }
      while ( v22 != v15 );
    }
  }
  return result;
}
