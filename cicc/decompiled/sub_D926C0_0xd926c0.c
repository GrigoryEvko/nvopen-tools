// Function: sub_D926C0
// Address: 0xd926c0
//
__int64 __fastcall sub_D926C0(__int64 *a1, __int64 *a2, _QWORD *a3)
{
  __int64 v4; // rbx
  __int16 v5; // ax
  __int64 result; // rax
  _QWORD *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rsi
  __int16 v10; // cx
  __int64 *v11; // rax
  __int64 *v12; // r13
  __int64 v13; // rdx
  bool v14; // zf
  __int64 v15; // rax
  __int64 v16; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v17[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = *a2;
  v5 = *(_WORD *)(*a2 + 24);
  if ( v5 != 6 )
  {
    if ( (unsigned __int16)(v5 - 9) > 3u )
      return 0;
    v11 = (__int64 *)*a1;
    v12 = a1;
    v13 = **(_QWORD **)(v4 + 32);
    v14 = *(_QWORD *)(*a1 + 16) == 0;
    v16 = v13;
    if ( !v14 )
    {
      a2 = &v16;
      a1 = v11;
      result = ((__int64 (__fastcall *)(__int64 *, __int64 *, _QWORD *))v11[3])(v11, &v16, a3);
      if ( (_BYTE)result )
        return result;
      v15 = *v12;
      v13 = *(_QWORD *)(*(_QWORD *)(v4 + 32) + 8LL);
      v14 = *(_QWORD *)(*v12 + 16) == 0;
      v17[0] = v13;
      if ( !v14 )
        return (*(__int64 (__fastcall **)(__int64, _QWORD *, _QWORD *))(v15 + 24))(v15, v17, a3);
    }
    sub_4263D6(a1, a2, v13);
  }
  result = 0;
  if ( *(_QWORD *)(v4 + 40) == 2 )
  {
    v7 = *(_QWORD **)(v4 + 32);
    v8 = *v7;
    v9 = v7[1];
    v10 = *(_WORD *)(*v7 + 24LL);
    if ( !v10 )
    {
      v10 = *(_WORD *)(v9 + 24);
      v8 = v7[1];
      v9 = *v7;
    }
    result = 0;
    if ( v10 == 7 && *(_QWORD *)(v8 + 40) == v9 )
    {
      *a3 = v9;
      return 1;
    }
  }
  return result;
}
