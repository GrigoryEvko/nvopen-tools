// Function: sub_62FF50
// Address: 0x62ff50
//
__int64 __fastcall sub_62FF50(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r14
  __int64 v3; // rax
  __int64 v4; // r12
  unsigned __int64 v6; // r13
  _QWORD *v7; // rax
  __int64 v8; // rdx

  v2 = (_QWORD *)a1;
  v3 = sub_724D50(10);
  *(_QWORD *)(v3 + 128) = a2;
  v4 = v3;
  if ( (unsigned int)sub_8D43F0(a2) )
  {
    if ( *(_BYTE *)(a1 + 173) != 10 )
      goto LABEL_3;
    v6 = 1;
  }
  else
  {
    v6 = sub_8D4490(a2);
    if ( v6 <= 1 )
    {
      v7 = (_QWORD *)a1;
      if ( *(_BYTE *)(a1 + 173) != 10 )
        goto LABEL_7;
    }
  }
  v7 = (_QWORD *)sub_724D50(11);
  v8 = *(_QWORD *)(a1 + 64);
  v7[23] = v6;
  v7[8] = v8;
  v7[22] = a1;
LABEL_7:
  if ( !v6 )
    return v4;
  v2 = v7;
LABEL_3:
  sub_72A690(v2, v4, 0, 0);
  return v4;
}
