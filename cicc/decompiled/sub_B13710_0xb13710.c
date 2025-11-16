// Function: sub_B13710
// Address: 0xb13710
//
_QWORD *__fastcall sub_B13710(__int64 a1)
{
  __int64 v1; // rsi
  _QWORD *result; // rax
  __int64 v3; // r15
  _QWORD *v4; // r12
  char v5; // cl
  __int64 v6; // rbx
  _QWORD *v7; // r15
  _QWORD *v8; // rax
  int v9; // ebx
  __int64 v10; // r14
  __int64 *v11; // rax
  __int64 *v12; // rdx
  char v13; // dl
  unsigned __int8 *v14; // rax
  __int64 v15; // [rsp+10h] [rbp-80h] BYREF
  _QWORD *v16; // [rsp+18h] [rbp-78h]
  __int64 v17; // [rsp+20h] [rbp-70h] BYREF
  __int64 *v18; // [rsp+28h] [rbp-68h]
  __int64 v19; // [rsp+30h] [rbp-60h]
  int v20; // [rsp+38h] [rbp-58h]
  char v21; // [rsp+3Ch] [rbp-54h]
  char v22; // [rsp+40h] [rbp-50h] BYREF

  v1 = a1;
  v17 = 0;
  v18 = (__int64 *)&v22;
  v19 = 4;
  v20 = 0;
  v21 = 1;
  result = sub_B129C0(&v15, a1);
  v3 = v15;
  v4 = v16;
  v5 = v21;
  if ( v16 != (_QWORD *)v15 )
  {
    do
    {
      v6 = v3;
      v7 = (_QWORD *)(v3 & 0xFFFFFFFFFFFFFFF8LL);
      v8 = v7;
      v9 = (v6 >> 2) & 1;
      if ( v9 )
        v8 = (_QWORD *)*v7;
      v10 = v8[17];
      if ( !v5 )
        goto LABEL_16;
      v11 = v18;
      v12 = &v18[HIDWORD(v19)];
      if ( v18 != v12 )
      {
        while ( v10 != *v11 )
        {
          if ( v12 == ++v11 )
            goto LABEL_18;
        }
        goto LABEL_9;
      }
LABEL_18:
      if ( HIDWORD(v19) < (unsigned int)v19 )
      {
        ++HIDWORD(v19);
        *v12 = v10;
        ++v17;
      }
      else
      {
LABEL_16:
        v1 = v10;
        sub_C8CC70(&v17, v10);
        v5 = v21;
        if ( !v13 )
          goto LABEL_9;
      }
      v14 = (unsigned __int8 *)sub_ACADE0(*(__int64 ***)(v10 + 8));
      v1 = v10;
      sub_B13360(a1, (unsigned __int8 *)v10, v14, 0);
      v5 = v21;
LABEL_9:
      if ( v9 || !v7 )
      {
        v3 = (unsigned __int64)(v7 + 1) | 4;
        result = (_QWORD *)v3;
      }
      else
      {
        result = v7 + 18;
        v3 = (__int64)(v7 + 18);
      }
    }
    while ( result != v4 );
  }
  if ( !v5 )
    return (_QWORD *)_libc_free(v18, v1);
  return result;
}
