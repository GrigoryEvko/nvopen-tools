// Function: sub_174B990
// Address: 0x174b990
//
_QWORD *__fastcall sub_174B990(__int64 ***a1, __int64 a2)
{
  __int64 v2; // rdx
  _QWORD *result; // rax
  __int64 **v4; // r13
  unsigned __int8 v5; // bl
  __int64 v6; // r14
  __int64 *v7; // r15
  unsigned __int8 *v8; // rax
  __int64 v9; // r12
  __int64 v10; // [rsp-68h] [rbp-68h]
  _QWORD *v11; // [rsp-60h] [rbp-60h]
  __int64 v12[2]; // [rsp-58h] [rbp-58h] BYREF
  __int16 v13; // [rsp-48h] [rbp-48h]

  v2 = (__int64)*(a1 - 3);
  result = 0;
  if ( *(_BYTE *)(v2 + 16) == 84 )
  {
    result = *(_QWORD **)(v2 + 8);
    if ( result )
    {
      result = (_QWORD *)result[1];
      if ( result )
      {
        return 0;
      }
      else
      {
        v4 = *a1;
        if ( *((_BYTE *)*a1 + 8) == 16 )
          v4 = (__int64 **)*(*a1)[2];
        if ( *(_BYTE *)(*(_QWORD *)(v2 - 72) + 16LL) == 9 )
        {
          v5 = *((_BYTE *)a1 + 16);
          v6 = *(_QWORD *)(v2 - 24);
          v10 = *(_QWORD *)(v2 - 48);
          v7 = (__int64 *)sub_1599EF0(*a1);
          v13 = 257;
          v8 = sub_1708970(a2, (unsigned int)v5 - 24, v10, v4, v12);
          v13 = 257;
          v9 = (__int64)v8;
          result = sub_1648A60(56, 3u);
          if ( result )
          {
            v11 = result;
            sub_15FA480((__int64)result, v7, v9, v6, (__int64)v12, 0);
            return v11;
          }
        }
      }
    }
  }
  return result;
}
