// Function: sub_7E02A0
// Address: 0x7e02a0
//
_QWORD *__fastcall sub_7E02A0(
        __int64 **a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        int a6,
        __int64 a7,
        _QWORD *a8,
        _QWORD *a9,
        _QWORD *a10)
{
  __int64 v10; // r14
  __int64 v12; // rbx
  char v13; // al
  __int64 v14; // rdi
  _QWORD *result; // rax
  _QWORD *v16; // r13
  __int64 v17; // rdx
  __int64 v18; // r12

  v10 = a3;
  *a9 = 0;
  if ( a3 != a2 )
  {
    v12 = a3;
    while ( 1 )
    {
      v13 = *(_BYTE *)(v12 + 96);
      if ( (v13 & 2) != 0 )
        break;
      if ( (v13 & 1) != 0 )
      {
        if ( !a2 )
        {
          result = 0;
          goto LABEL_19;
        }
        if ( !a6 )
          BUG();
        v12 = 0;
LABEL_12:
        v14 = *(_QWORD *)(a2 + 104);
LABEL_13:
        if ( a4 )
          v14 -= *(_QWORD *)(a4 + 104);
        if ( !(v14 | a5) )
        {
          v10 = a4;
LABEL_17:
          result = 0;
          if ( a2 )
            goto LABEL_18;
LABEL_19:
          *a8 = result;
          if ( v10 )
          {
            result = (_QWORD *)((char *)result - *(_QWORD *)(v10 + 104));
            *a8 = result;
          }
          return result;
        }
        if ( !a5 )
        {
          if ( a7 )
          {
            result = *(_QWORD **)(a7 + 112);
            if ( result )
            {
              while ( 1 )
              {
                if ( result[34] != a7 )
                  goto LABEL_22;
                if ( !result[37] && result[36] == v14 && !result[38] )
                  break;
                result = (_QWORD *)result[14];
                if ( !result )
                  goto LABEL_22;
              }
              *a8 = v14;
              *a9 = 0;
              if ( a10 )
                *a10 = result;
              return result;
            }
          }
        }
LABEL_22:
        v16 = *(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v12 + 40) + 168LL) + 64LL);
        v17 = **a1;
        while ( 1 )
        {
          v18 = v16[1];
          if ( *(_QWORD *)v18 && (**(_QWORD **)v18 == v17 || *(_BYTE *)(v18 + 174) == 2 && *((_BYTE *)a1 + 174) == 2) )
          {
            if ( (unsigned int)sub_8DE890(*(_QWORD *)(v18 + 152), a1[19], 0, 0)
              && (unsigned int)sub_8D7820(*(_QWORD *)(v18 + 152), a1[19], 0, 0) )
            {
              a2 = v12;
              *a9 = v16[3];
              goto LABEL_18;
            }
            v17 = **a1;
          }
          v16 = (_QWORD *)*v16;
        }
      }
      v12 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v12 + 112) + 16LL) + 8LL) + 16LL);
      if ( a2 == v12 )
        goto LABEL_17;
    }
    if ( a6 )
    {
      v14 = 0;
      if ( a2 )
        goto LABEL_12;
      goto LABEL_13;
    }
    goto LABEL_22;
  }
  if ( a3 )
  {
LABEL_18:
    result = *(_QWORD **)(a2 + 104);
    goto LABEL_19;
  }
  *a8 = 0;
  return a8;
}
