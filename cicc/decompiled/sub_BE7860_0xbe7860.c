// Function: sub_BE7860
// Address: 0xbe7860
//
__int64 __fastcall sub_BE7860(__int64 **a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v6; // rdx
  __int64 result; // rax
  __int64 *v8; // rbx
  __int64 *v9; // r13
  __int64 v10; // r12
  _BYTE *v11; // rax
  _BYTE *v12; // rsi
  _BYTE *v13; // rdi
  __int64 v14; // rdi
  _BYTE *v15; // rax
  __int64 *v16; // r9
  __int64 *v17; // rdx
  _QWORD v18[2]; // [rsp+0h] [rbp-80h] BYREF
  __int64 v19; // [rsp+10h] [rbp-70h]
  __int64 v20; // [rsp+18h] [rbp-68h]
  __int16 v21; // [rsp+20h] [rbp-60h]
  _QWORD v22[2]; // [rsp+30h] [rbp-50h] BYREF
  const char *v23; // [rsp+40h] [rbp-40h]
  __int16 v24; // [rsp+50h] [rbp-30h]

  v6 = **a1;
  if ( a4 >= *(_DWORD *)(v6 + 12) - 1 )
  {
    v16 = a1[2];
    v20 = a3;
    v18[0] = "'allocsize' ";
    v17 = a1[1];
    v22[0] = v18;
    v24 = 770;
    v23 = " argument is out of bounds";
    v21 = 1283;
    v19 = a2;
    sub_BE7760(v16, (__int64)v22, (_BYTE **)v17);
    return 0;
  }
  else
  {
    result = 1;
    if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v6 + 16) + 8LL * (a4 + 1)) + 8LL) == 12 )
      return result;
    v8 = a1[2];
    v9 = a1[1];
    v19 = a2;
    v21 = 1283;
    v18[0] = "'allocsize' ";
    v22[0] = v18;
    v20 = a3;
    v23 = " argument must refer to an integer parameter";
    v24 = 770;
    v10 = *v8;
    if ( *v8 )
    {
      sub_CA0E80(v22, *v8);
      v11 = *(_BYTE **)(v10 + 32);
      if ( (unsigned __int64)v11 >= *(_QWORD *)(v10 + 24) )
      {
        sub_CB5D20(v10, 10);
      }
      else
      {
        *(_QWORD *)(v10 + 32) = v11 + 1;
        *v11 = 10;
      }
      v12 = (_BYTE *)*v8;
      *((_BYTE *)v8 + 152) = 1;
      if ( !v12 )
        return 0;
      v13 = (_BYTE *)*v9;
      if ( !*v9 )
        return 0;
      if ( *v13 <= 0x1Cu )
      {
        sub_A5C020(v13, (__int64)v12, 1, (__int64)(v8 + 2));
        v14 = *v8;
        v15 = *(_BYTE **)(*v8 + 32);
        if ( (unsigned __int64)v15 >= *(_QWORD *)(*v8 + 24) )
          goto LABEL_10;
      }
      else
      {
        sub_A693B0((__int64)v13, v12, (__int64)(v8 + 2), 0);
        v14 = *v8;
        v15 = *(_BYTE **)(*v8 + 32);
        if ( (unsigned __int64)v15 >= *(_QWORD *)(*v8 + 24) )
        {
LABEL_10:
          sub_CB5D20(v14, 10);
          return 0;
        }
      }
      *(_QWORD *)(v14 + 32) = v15 + 1;
      *v15 = 10;
      return 0;
    }
    else
    {
      *((_BYTE *)v8 + 152) = 1;
      return 0;
    }
  }
}
