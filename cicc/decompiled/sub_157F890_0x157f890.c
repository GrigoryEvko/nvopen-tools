// Function: sub_157F890
// Address: 0x157f890
//
__int64 __fastcall sub_157F890(_QWORD *a1, _QWORD *a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 result; // rax
  __int64 v7; // r14
  _QWORD *v8; // rbx
  __int64 v9; // rax
  _QWORD *v10; // rbx

  v4 = sub_157E9B0((__int64)(a1 - 5));
  *a2 = a3;
  v5 = v4;
  result = sub_157E9B0((__int64)(a1 - 5));
  if ( v5 != result )
  {
    v7 = result;
    result = *a1 & 0xFFFFFFFFFFFFFFF8LL;
    if ( a1 != (_QWORD *)result )
    {
      if ( v5 )
      {
        v8 = (_QWORD *)a1[1];
        if ( a1 != v8 )
        {
          while ( v8 )
          {
            if ( (*((_BYTE *)v8 - 1) & 0x20) != 0 )
            {
              v9 = sub_16498B0(v8 - 3);
              result = sub_164D860(v5, v9);
            }
            v8 = (_QWORD *)v8[1];
            if ( a1 == v8 )
              goto LABEL_9;
          }
LABEL_16:
          BUG();
        }
      }
      else
      {
LABEL_9:
        if ( v7 )
        {
          v10 = (_QWORD *)a1[1];
          if ( a1 != v10 )
          {
            while ( v10 )
            {
              if ( (*((_BYTE *)v10 - 1) & 0x20) != 0 )
                result = sub_164D6D0(v7, v10 - 3);
              v10 = (_QWORD *)v10[1];
              if ( a1 == v10 )
                return result;
            }
            goto LABEL_16;
          }
        }
      }
    }
  }
  return result;
}
