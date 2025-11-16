// Function: sub_806430
// Address: 0x806430
//
_QWORD *__fastcall sub_806430(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int *a5)
{
  __int64 *v8; // r12
  _QWORD *result; // rax
  __int64 v10; // r15
  __int64 v11; // rdi
  __int16 v12; // dx
  __int64 *v13; // rax
  _BYTE *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // r9
  __int64 v20; // rax
  __int64 v22; // [rsp+10h] [rbp-40h]
  __m128i *v23; // [rsp+18h] [rbp-38h]

  v8 = *(__int64 **)(a1 + 168);
  result = sub_7F5B50(a1, a2, a3, a4, a5);
  v10 = *v8;
  if ( *v8 )
  {
    v23 = 0;
    while ( 1 )
    {
      v12 = *(_WORD *)(v10 + 136);
      if ( v12 )
      {
        v13 = sub_7FCC60(a3, 0, v12);
        v14 = sub_73DCD0(v13);
        result = sub_731370((__int64)v14, 0, v15, v16, v17, v18);
        v19 = (__int64)result;
      }
      else
      {
        if ( *(_QWORD *)(v10 + 128) == -1 )
          goto LABEL_5;
        v11 = v8[24];
        if ( !v11 )
          goto LABEL_5;
        result = (_QWORD *)v8[3];
        if ( result )
        {
          while ( result != (_QWORD *)v10 )
          {
            result = (_QWORD *)result[3];
            if ( !result )
              goto LABEL_16;
          }
          goto LABEL_5;
        }
LABEL_16:
        result = sub_7F5690(v11, a1, v10);
        v19 = (__int64)result;
      }
      if ( v19 )
      {
        v22 = v19;
        if ( a4 )
        {
          v20 = sub_7F5750(a4, (_QWORD *)v10);
          result = sub_7F5940(v22, v23->m128i_i64, v20, a5);
        }
        else
        {
          v23 = sub_7E8890(a2, v10, 0);
          result = sub_7F5940(v22, v23->m128i_i64, 0, a5);
        }
        v10 = *(_QWORD *)v10;
        if ( !v10 )
          return result;
      }
      else
      {
LABEL_5:
        v10 = *(_QWORD *)v10;
        if ( !v10 )
          return result;
      }
    }
  }
  return result;
}
