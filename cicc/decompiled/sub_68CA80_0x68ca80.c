// Function: sub_68CA80
// Address: 0x68ca80
//
__int64 __fastcall sub_68CA80(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // rdx
  int v4; // eax
  unsigned int v5; // r12d
  __int64 v6; // rdx
  __int64 v7; // rsi
  __int64 *v8; // rax
  __int64 *v9; // r12
  __int64 v10; // rax
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 *v15; // rax
  __int64 *v16; // r12
  __int64 v17; // rax
  _QWORD v18[5]; // [rsp+8h] [rbp-28h] BYREF

  v18[0] = 0;
  while ( 1 )
  {
    while ( 1 )
    {
      result = *(unsigned __int8 *)(a1 + 24);
      if ( (_BYTE)result != 1 )
      {
        while ( (_BYTE)result == 5 )
        {
          if ( (unsigned int)sub_6EC5C0(*(_QWORD *)a1, v18) )
          {
            v5 = 3005;
            if ( v18[0] )
              goto LABEL_17;
            v5 = 3004;
            goto LABEL_27;
          }
          v3 = *(_QWORD *)(a1 + 56);
          result = *(unsigned __int8 *)(v3 + 48);
          if ( (_BYTE)result == 5 )
          {
            v7 = *(_QWORD *)(v3 + 56);
            if ( !v7 || (*(_BYTE *)(v7 + 196) & 0x20) == 0 )
              return result;
            v8 = sub_5CF860(12, v7);
            v9 = v8;
            if ( v8 && (unsigned int)sub_72AE80(*(_QWORD *)(v8[4] + 40)) )
            {
              v10 = v9[4];
              v5 = 3003;
              v6 = *(_QWORD *)(*(_QWORD *)(v10 + 40) + 184LL);
              v18[0] = v6;
            }
            else
            {
              v6 = v18[0];
              v5 = 3002;
            }
            goto LABEL_24;
          }
          result = (unsigned int)(result - 3);
          if ( (unsigned __int8)result > 1u )
            return result;
          a1 = *(_QWORD *)(v3 + 56);
          result = *(unsigned __int8 *)(a1 + 24);
          if ( (_BYTE)result == 1 )
            goto LABEL_8;
        }
        return result;
      }
LABEL_8:
      v4 = *(unsigned __int8 *)(a1 + 56);
      if ( (_BYTE)v4 == 5 )
      {
        result = sub_8D2600(*(_QWORD *)a1);
        if ( (_DWORD)result )
          return result;
        goto LABEL_12;
      }
      if ( (_BYTE)v4 != 91 )
        break;
      a1 = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 16LL);
    }
    if ( (_BYTE)v4 == 103 )
    {
      sub_68CA80(*(_QWORD *)(*(_QWORD *)(a1 + 72) + 16LL));
      return sub_68CA80(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 72) + 16LL) + 16LL));
    }
    if ( (_BYTE)v4 != 4 )
      break;
LABEL_12:
    a1 = *(_QWORD *)(a1 + 72);
  }
  result = (unsigned int)(v4 - 105);
  if ( (unsigned __int8)result > 4u )
    return result;
  v11 = sub_6EE7B0(a1);
  v12 = sub_72B0F0(*(_QWORD *)(a1 + 72), 0);
  v13 = v12;
  if ( v12
    && (*(_BYTE *)(v12 + 196) & 0x20) != 0
    && !(unsigned int)sub_8D2600(*(_QWORD *)(*(_QWORD *)(v12 + 152) + 160LL)) )
  {
    v15 = sub_5CF860(12, v13);
    v16 = v15;
    if ( v15 && (unsigned int)sub_72AE80(*(_QWORD *)(v15[4] + 40)) )
    {
      v17 = v16[4];
      v5 = 3000;
      v6 = *(_QWORD *)(*(_QWORD *)(v17 + 40) + 184LL);
      v18[0] = v6;
    }
    else
    {
      v6 = v18[0];
      v5 = 2809;
    }
LABEL_24:
    if ( !v6 )
    {
LABEL_27:
      result = sub_6E53E0(5, v5, a1 + 28);
      if ( (_DWORD)result )
        return sub_684B30(v5, (_DWORD *)(a1 + 28));
      return result;
    }
    return sub_6E5C20(v5, a1 + 28, v6);
  }
  result = sub_8D2310(v11);
  if ( (_DWORD)result )
  {
    v14 = sub_73D790(v11);
    result = sub_6EC5C0(v14, v18);
    if ( (_DWORD)result )
    {
      v5 = 3001;
      if ( !v18[0] )
      {
        v5 = 2810;
        goto LABEL_27;
      }
LABEL_17:
      v6 = v18[0];
      return sub_6E5C20(v5, a1 + 28, v6);
    }
  }
  return result;
}
