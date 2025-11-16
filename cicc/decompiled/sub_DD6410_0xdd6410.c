// Function: sub_DD6410
// Address: 0xdd6410
//
__int64 *__fastcall sub_DD6410(_QWORD *a1, __int64 a2)
{
  _QWORD *v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // rdi
  int v8; // eax
  int v9; // r8d
  unsigned int v10; // ecx
  __int64 *v11; // rax
  __int64 v12; // r9
  _QWORD *v13; // rax
  __int64 *result; // rax
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // rsi
  int v20; // eax
  int v21; // r10d
  _QWORD v22[8]; // [rsp+0h] [rbp-60h] BYREF
  __int16 v23; // [rsp+40h] [rbp-20h]

  v4 = (_QWORD *)a1[194];
  if ( v4 )
  {
    v5 = a1[6];
    v6 = *(_QWORD *)(a2 + 40);
    v7 = *(_QWORD *)(v5 + 8);
    v8 = *(_DWORD *)(v5 + 24);
    if ( v8 )
    {
      v9 = v8 - 1;
      v10 = (v8 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v11 = (__int64 *)(v7 + 16LL * v10);
      v12 = *v11;
      if ( v6 == *v11 )
      {
LABEL_4:
        v13 = (_QWORD *)v11[1];
        if ( v13 && v4 != v13 )
        {
          while ( 1 )
          {
            v13 = (_QWORD *)*v13;
            if ( v4 == v13 )
              break;
            if ( !v13 )
              return sub_DA3860(a1, a2);
          }
        }
      }
      else
      {
        v20 = 1;
        while ( v12 != -4096 )
        {
          v21 = v20 + 1;
          v10 = v9 & (v20 + v10);
          v11 = (__int64 *)(v7 + 16LL * v10);
          v12 = *v11;
          if ( v6 == *v11 )
            goto LABEL_4;
          v20 = v21;
        }
      }
    }
  }
  result = sub_DD5A30((__int64)a1, a2);
  if ( !result )
  {
    v15 = a1[1];
    v16 = a1[4];
    v22[2] = 0;
    v17 = a1[3];
    v18 = a1[5];
    memset(&v22[5], 0, 24);
    v22[0] = v15;
    v22[4] = v16;
    v22[1] = v17;
    v22[3] = v18;
    v23 = 1;
    v19 = ((__int64 (__fastcall *)(__int64, _QWORD *))sub_1020E10)(a2, v22);
    if ( v19 )
    {
      return (__int64 *)sub_DD8400(a1, v19);
    }
    else
    {
      result = (__int64 *)sub_DE0740(a1, a2);
      if ( !result )
      {
        result = (__int64 *)sub_DD9AA0(a1, a2);
        if ( !result )
          return sub_DA3860(a1, a2);
      }
    }
  }
  return result;
}
