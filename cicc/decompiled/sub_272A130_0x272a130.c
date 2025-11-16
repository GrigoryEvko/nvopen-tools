// Function: sub_272A130
// Address: 0x272a130
//
void __fastcall sub_272A130(unsigned __int8 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 i; // r14
  __int64 v7; // rsi
  __int64 *v8; // rax
  __int64 *v9; // rdx
  __int64 v10; // rax
  __int64 v11; // r15
  __int64 *v12; // rdx
  __int64 v13; // r8
  __int64 v14; // r9
  unsigned __int8 v15; // cl
  __int64 *v16; // rax
  __int64 v17; // [rsp+10h] [rbp-70h] BYREF
  __int64 *v18; // [rsp+18h] [rbp-68h]
  __int64 v19; // [rsp+20h] [rbp-60h]
  int v20; // [rsp+28h] [rbp-58h]
  unsigned __int8 v21; // [rsp+2Ch] [rbp-54h]
  char v22; // [rsp+30h] [rbp-50h] BYREF

  v17 = 0;
  v18 = (__int64 *)&v22;
  v19 = 4;
  v20 = 0;
  v21 = 1;
  if ( a2 == a4 )
    return;
  for ( i = a2; ; i = v11 )
  {
    v7 = sub_AA54C0(i);
    if ( v21 )
      break;
    if ( sub_C8CA60((__int64)&v17, v7) )
      goto LABEL_23;
LABEL_10:
    v10 = sub_AA54C0(i);
    v11 = v10;
    if ( !v10 )
    {
LABEL_23:
      v15 = v21;
LABEL_19:
      if ( !v15 )
        _libc_free((unsigned __int64)v18);
      return;
    }
    sub_2729E70(a1, v10, i, a3);
    v15 = v21;
    if ( v21 )
    {
      v16 = v18;
      v12 = &v18[HIDWORD(v19)];
      if ( v18 != v12 )
      {
        while ( v11 != *v16 )
        {
          if ( v12 == ++v16 )
            goto LABEL_21;
        }
LABEL_16:
        if ( a4 == v11 )
          goto LABEL_19;
        continue;
      }
LABEL_21:
      if ( HIDWORD(v19) < (unsigned int)v19 )
      {
        ++HIDWORD(v19);
        *v12 = v11;
        v15 = v21;
        ++v17;
        goto LABEL_16;
      }
    }
    sub_C8CC70((__int64)&v17, v11, (__int64)v12, v21, v13, v14);
    v15 = v21;
    if ( a4 == v11 )
      goto LABEL_19;
  }
  v8 = v18;
  v9 = &v18[HIDWORD(v19)];
  if ( v18 == v9 )
    goto LABEL_10;
  while ( v7 != *v8 )
  {
    if ( v9 == ++v8 )
      goto LABEL_10;
  }
}
