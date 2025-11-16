// Function: sub_2901070
// Address: 0x2901070
//
__int64 __fastcall sub_2901070(char *a1, __int64 *a2, __int64 a3)
{
  unsigned __int8 v3; // al
  __int64 *v4; // rbx
  __int64 v5; // r12
  bool v6; // zf
  __int64 v7; // rsi
  __int64 result; // rax
  __int64 *v9; // r13
  __int64 v10; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v11[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = *a1;
  if ( (unsigned __int8)*a1 <= 0x1Cu )
    goto LABEL_24;
  v4 = (__int64 *)a1;
  v5 = (__int64)a2;
  if ( v3 != 84 )
  {
    if ( v3 == 86 )
    {
      sub_2901040((__int64)a2, *((_QWORD *)a1 - 8), a3);
LABEL_9:
      v7 = *((_QWORD *)a1 - 4);
      return sub_2901040(v5, v7, a3);
    }
    if ( v3 != 90 )
    {
      if ( v3 != 91 )
      {
        if ( v3 == 92 )
        {
          v6 = a2[2] == 0;
          v11[0] = *((_QWORD *)a1 - 8);
          if ( !v6 )
          {
            ((void (__fastcall *)(__int64 *, _QWORD *))a2[3])(a2, v11);
            a3 = *((unsigned int *)a1 + 20);
            if ( *(_DWORD *)(*(_QWORD *)(*((_QWORD *)a1 - 8) + 8LL) + 32LL) == (_DWORD)a3 )
            {
              result = sub_B4EE20(*((int **)a1 + 9), (unsigned int)a3, a3);
              if ( (_BYTE)result )
                return result;
            }
            goto LABEL_9;
          }
LABEL_23:
          sub_4263D6(a1, a2, a3);
        }
LABEL_24:
        BUG();
      }
      sub_2901040((__int64)a2, *((_QWORD *)a1 - 12), a3);
    }
    v7 = *((_QWORD *)a1 - 8);
    return sub_2901040(v5, v7, a3);
  }
  result = 32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF);
  if ( (a1[7] & 0x40) != 0 )
  {
    v9 = (__int64 *)*((_QWORD *)a1 - 1);
    v4 = (__int64 *)((char *)v9 + result);
  }
  else
  {
    v9 = (__int64 *)&a1[-result];
  }
  if ( v4 != v9 )
  {
    while ( 1 )
    {
      v6 = *(_QWORD *)(v5 + 16) == 0;
      v10 = *v9;
      if ( v6 )
        break;
      v9 += 4;
      a2 = &v10;
      a1 = (char *)v5;
      result = (*(__int64 (__fastcall **)(__int64, __int64 *))(v5 + 24))(v5, &v10);
      if ( v4 == v9 )
        return result;
    }
    goto LABEL_23;
  }
  return result;
}
