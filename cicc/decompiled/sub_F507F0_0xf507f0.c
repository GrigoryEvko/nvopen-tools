// Function: sub_F507F0
// Address: 0xf507f0
//
__int64 __fastcall sub_F507F0(__int64 a1)
{
  __int64 v1; // rsi
  __int64 *v2; // rdx
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 result; // rax
  __int64 v6; // r12
  __int64 v7; // rcx
  __int64 v8; // rbx
  _QWORD *v9; // rax
  _QWORD *v10; // r14
  int v11; // ebx
  __int64 v12; // r15
  __int64 *v13; // rax
  unsigned __int8 *v14; // rax
  __int64 v15; // [rsp+10h] [rbp-80h] BYREF
  __int64 v16; // [rsp+18h] [rbp-78h]
  __int64 v17; // [rsp+20h] [rbp-70h] BYREF
  __int64 *v18; // [rsp+28h] [rbp-68h]
  __int64 v19; // [rsp+30h] [rbp-60h]
  int v20; // [rsp+38h] [rbp-58h]
  unsigned __int8 v21; // [rsp+3Ch] [rbp-54h]
  char v22; // [rsp+40h] [rbp-50h] BYREF

  v1 = a1;
  v18 = (__int64 *)&v22;
  v17 = 0;
  v19 = 4;
  v20 = 0;
  v21 = 1;
  sub_B58E30(&v15, a1);
  result = v15;
  v6 = v16;
  v7 = v21;
  if ( v16 != v15 )
  {
    do
    {
      v8 = result;
      v9 = (_QWORD *)(result & 0xFFFFFFFFFFFFFFF8LL);
      v10 = v9;
      v11 = (v8 >> 2) & 1;
      if ( v11 )
        v9 = (_QWORD *)*v9;
      v12 = v9[17];
      if ( !(_BYTE)v7 )
        goto LABEL_14;
      v13 = v18;
      v1 = HIDWORD(v19);
      v2 = &v18[HIDWORD(v19)];
      if ( v18 != v2 )
      {
        while ( v12 != *v13 )
        {
          if ( v2 == ++v13 )
            goto LABEL_17;
        }
LABEL_9:
        if ( v11 )
          goto LABEL_16;
        goto LABEL_10;
      }
LABEL_17:
      if ( HIDWORD(v19) < (unsigned int)v19 )
      {
        ++HIDWORD(v19);
        *v2 = v12;
        ++v17;
      }
      else
      {
LABEL_14:
        v1 = v12;
        sub_C8CC70((__int64)&v17, v12, (__int64)v2, v7, v3, v4);
        v7 = v21;
        if ( !(_BYTE)v2 )
          goto LABEL_9;
      }
      v14 = (unsigned __int8 *)sub_ACADE0(*(__int64 ***)(v12 + 8));
      v1 = v12;
      sub_B59720(a1, v12, v14);
      v7 = v21;
      if ( v11 )
      {
LABEL_16:
        result = (unsigned __int64)(v10 + 1) | 4;
        v3 = result;
        continue;
      }
LABEL_10:
      v3 = (__int64)(v10 + 18);
      result = (__int64)(v10 + 18);
    }
    while ( v3 != v6 );
  }
  if ( !(_BYTE)v7 )
    return _libc_free(v18, v1);
  return result;
}
