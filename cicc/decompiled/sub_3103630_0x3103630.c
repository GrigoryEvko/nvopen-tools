// Function: sub_3103630
// Address: 0x3103630
//
__int64 __fastcall sub_3103630(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  _QWORD *v9; // rdx
  __int64 *v10; // rax
  unsigned __int64 v11; // r13
  __int64 *v12; // rbx
  __int64 v13; // r12
  __int64 *v14; // rax
  __int64 v15; // rsi
  unsigned __int8 v16; // [rsp+Fh] [rbp-61h]
  __int64 v17; // [rsp+10h] [rbp-60h] BYREF
  __int64 *v18; // [rsp+18h] [rbp-58h]
  __int64 v19; // [rsp+20h] [rbp-50h]
  int v20; // [rsp+28h] [rbp-48h]
  char v21; // [rsp+2Ch] [rbp-44h]
  char v22; // [rsp+30h] [rbp-40h] BYREF

  result = 1;
  v9 = *(_QWORD **)(a3 + 32);
  if ( a2 != *v9 )
  {
    v21 = 1;
    v17 = 0;
    v18 = (__int64 *)&v22;
    v19 = 4;
    v20 = 0;
    if ( a2 != *v9 )
    {
      sub_3102230(a3, a2, (__int64)&v17, a4, (__int64)&v17, a6);
      v10 = v18;
      v11 = (unsigned __int64)(v21 ? &v18[HIDWORD(v19)] : &v18[(unsigned int)v19]);
      if ( v18 != (__int64 *)v11 )
      {
        while ( 1 )
        {
          v12 = v10;
          if ( (unsigned __int64)*v10 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( (__int64 *)v11 == ++v10 )
            goto LABEL_8;
        }
        if ( (__int64 *)v11 != v10 )
        {
          v13 = a1 + 88;
          if ( sub_30ED150(v13, *v10) )
          {
LABEL_20:
            result = 0;
LABEL_9:
            if ( !v21 )
            {
              v16 = result;
              _libc_free((unsigned __int64)v18);
              return v16;
            }
            return result;
          }
          while ( 1 )
          {
            v14 = v12 + 1;
            if ( v12 + 1 == (__int64 *)v11 )
              break;
            v15 = *v14;
            for ( ++v12; (unsigned __int64)*v14 >= 0xFFFFFFFFFFFFFFFELL; v12 = v14 )
            {
              if ( (__int64 *)v11 == ++v14 )
                goto LABEL_8;
              v15 = *v14;
            }
            if ( (__int64 *)v11 == v12 )
              break;
            if ( sub_30ED150(v13, v15) )
              goto LABEL_20;
          }
        }
      }
    }
LABEL_8:
    result = 1;
    goto LABEL_9;
  }
  return result;
}
