// Function: sub_199CCC0
// Address: 0x199ccc0
//
__int64 __fastcall sub_199CCC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5, __int64 a6, __int64 a7)
{
  _QWORD *v8; // r12
  __int16 v9; // ax
  _QWORD *v11; // rdx
  _QWORD *v12; // rax
  int v13; // eax
  __int64 *v14; // rax
  __int16 v15; // ax
  __int64 result; // rax
  __int16 v17; // ax
  __int64 *v18; // rax
  __int64 v19; // [rsp-10h] [rbp-80h]
  _BYTE v21[96]; // [rsp+10h] [rbp-60h] BYREF

  v8 = a5;
  v9 = *(_WORD *)(a2 + 24);
  if ( v9 != 7 )
    goto LABEL_13;
  v11 = *(_QWORD **)(a2 + 48);
  if ( v11 == a5 )
  {
LABEL_7:
    v13 = !(unsigned __int8)sub_14A2B00(a7)
       || *(_WORD *)(sub_13A5BC0((_QWORD *)a2, a6) + 24)
       || (sub_1456040(**(_QWORD **)(a2 + 32)), !(unsigned __int8)sub_14A3850(a7))
       && (sub_1456040(**(_QWORD **)(a2 + 32)), !(unsigned __int8)sub_14A3880(a7))
       || (v18 = *(__int64 **)(a2 + 32), !*(_WORD *)(*v18 + 24))
       || !sub_146CEE0(a6, *v18, (__int64)v8);
    *(_DWORD *)(a1 + 8) += v13;
    v14 = *(__int64 **)(a2 + 32);
    if ( (*(_QWORD *)(a2 + 40) != 2 || *(_WORD *)(v14[1] + 24)) && !sub_199CBE0(a3, *v14) )
    {
      sub_199AF80((__int64)v21, a4, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
      if ( v21[32] )
      {
        sub_199CCC0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL), a3, a4, (_DWORD)v8, a6, a7);
        result = v19;
        if ( *(_DWORD *)(a1 + 4) == -1 )
          return result;
      }
    }
    v9 = *(_WORD *)(a2 + 24);
LABEL_13:
    if ( v9 )
    {
      ++*(_DWORD *)(a1 + 4);
      v15 = *(_WORD *)(a2 + 24);
      if ( v15 && v15 != 10 )
      {
        if ( v15 == 7 )
        {
          v17 = *(_WORD *)(**(_QWORD **)(a2 + 32) + 24LL);
          if ( v17 == 10 || !v17 )
            goto LABEL_19;
        }
        ++*(_DWORD *)(a1 + 24);
        v15 = *(_WORD *)(a2 + 24);
      }
      if ( v15 == 5 )
      {
        result = sub_146D100(a6, a2, (__int64)v8);
        goto LABEL_20;
      }
    }
LABEL_19:
    result = 0;
LABEL_20:
    *(_DWORD *)(a1 + 12) += result;
    return result;
  }
  if ( a5 )
  {
    v12 = a5;
    do
    {
      v12 = (_QWORD *)*v12;
      if ( v11 == v12 )
        goto LABEL_7;
    }
    while ( v12 );
  }
  result = sub_1993E40(a2, a6);
  if ( !(_BYTE)result )
  {
    result = *(_QWORD *)(a2 + 48);
    if ( v8 == (_QWORD *)result )
    {
LABEL_39:
      ++*(_DWORD *)(a1 + 4);
    }
    else
    {
      while ( v8 )
      {
        v8 = (_QWORD *)*v8;
        if ( (_QWORD *)result == v8 )
          goto LABEL_39;
      }
      *(_QWORD *)a1 = -1;
      *(_QWORD *)(a1 + 8) = -1;
      *(_QWORD *)(a1 + 16) = -1;
      *(_QWORD *)(a1 + 24) = -1;
    }
  }
  return result;
}
