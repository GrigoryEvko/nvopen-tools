// Function: sub_26991F0
// Address: 0x26991f0
//
__int64 __fastcall sub_26991F0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  unsigned __int8 *v4; // rax
  unsigned __int8 *v5; // rax
  __int64 v6; // rsi
  __int64 v7; // rax
  __int64 v8; // rdi
  unsigned __int8 *v9; // rax
  _BYTE *v10; // r12
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdi
  char v14; // r13
  unsigned __int8 *v15; // rax
  unsigned int v16; // [rsp-30h] [rbp-30h] BYREF
  _DWORD v17[11]; // [rsp-2Ch] [rbp-2Ch] BYREF

  if ( !*(_QWORD *)(a1 + 296) )
    return 1;
  result = 1;
  if ( *(_QWORD *)(a1 + 312) )
  {
    v16 = 1;
    if ( (unsigned __int8)sub_26968A0(a1, a2, &v16) )
    {
      v4 = (unsigned __int8 *)sub_2674090(*(_QWORD *)(a1 + 296), a2);
      sub_2673FE0(v4);
      goto LABEL_5;
    }
    v7 = *(_QWORD *)(a1 + 296);
    v8 = *(_QWORD *)(v7 - 32);
    if ( v8 )
    {
      if ( *(_BYTE *)v8 )
      {
        v8 = 0;
      }
      else if ( *(_QWORD *)(v8 + 24) != *(_QWORD *)(v7 + 80) )
      {
        v8 = 0;
      }
    }
    if ( sub_B2FC80(v8) )
    {
      v9 = (unsigned __int8 *)sub_2674090(*(_QWORD *)(a1 + 296), a2);
      v10 = sub_2673FE0(v9);
    }
    else
    {
      v14 = sub_2678420(a1, a2, &v16);
      v15 = (unsigned __int8 *)sub_2674090(*(_QWORD *)(a1 + 296), a2);
      v10 = sub_2673FE0(v15);
      if ( v14 )
      {
LABEL_5:
        v5 = sub_2674070(*(_QWORD *)(a1 + 296), a2);
        v6 = *(_QWORD *)(a1 + 304);
        if ( *((_QWORD *)v5 - 4) == v6 )
          return v16;
        sub_B30160((__int64)v5, v6);
        return 0;
      }
    }
    v11 = sub_2673FD0(*(unsigned __int8 **)(a1 + 304));
    v17[0] = 0;
    v12 = sub_AAAE30(v11, (__int64)v10, v17, 1);
    v13 = *(_QWORD *)(a1 + 304);
    a2 = v12;
    v17[0] = 0;
    *(_QWORD *)(a1 + 304) = sub_AAAE30(v13, v12, v17, 1);
    goto LABEL_5;
  }
  return result;
}
