// Function: sub_1C51230
// Address: 0x1c51230
//
__int64 __fastcall sub_1C51230(__int64 *a1, unsigned __int64 a2)
{
  __int64 result; // rax
  __int64 *v3; // r15
  __int64 v4; // r14
  _QWORD *v5; // r12
  __int64 v6; // rbx
  unsigned __int64 v7; // rdx
  _BOOL4 v8; // r9d
  __int64 v9; // rax
  _BOOL4 v10; // [rsp+4h] [rbp-3Ch]
  __int64 *v11; // [rsp+8h] [rbp-38h]

  result = *a1;
  v3 = *(__int64 **)(*a1 + 32);
  v11 = *(__int64 **)(*a1 + 40);
  if ( v3 != v11 )
  {
    while ( 1 )
    {
      v4 = *v3;
      v5 = *(_QWORD **)(*v3 + 72);
      v6 = *v3 + 64;
      if ( !v5 )
        break;
      while ( 1 )
      {
        v7 = v5[4];
        result = v5[3];
        if ( a2 < v7 )
          result = v5[2];
        if ( !result )
          break;
        v5 = (_QWORD *)result;
      }
      if ( a2 < v7 )
      {
        if ( *(_QWORD **)(v4 + 80) != v5 )
          goto LABEL_15;
LABEL_10:
        v8 = 1;
        if ( (_QWORD *)v6 == v5 )
        {
LABEL_11:
          v10 = v8;
          v9 = sub_22077B0(40);
          *(_QWORD *)(v9 + 32) = a2;
          result = sub_220F040(v10, v9, v5, v6);
          ++*(_QWORD *)(v4 + 96);
          goto LABEL_12;
        }
LABEL_17:
        v8 = a2 < v5[4];
        goto LABEL_11;
      }
      if ( v7 < a2 )
        goto LABEL_10;
LABEL_12:
      if ( v11 == ++v3 )
        return result;
    }
    v5 = (_QWORD *)(*v3 + 64);
    if ( v6 == *(_QWORD *)(v4 + 80) )
    {
      v8 = 1;
      goto LABEL_11;
    }
LABEL_15:
    result = sub_220EF80(v5);
    if ( a2 <= *(_QWORD *)(result + 32) )
      goto LABEL_12;
    v8 = 1;
    if ( (_QWORD *)v6 == v5 )
      goto LABEL_11;
    goto LABEL_17;
  }
  return result;
}
