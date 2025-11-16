// Function: sub_3280A00
// Address: 0x3280a00
//
char __fastcall sub_3280A00(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int16 v3; // bx
  __int64 v4; // rax
  unsigned __int64 v5; // r13
  __int64 v6; // rdx
  char v7; // r14
  __int64 v8; // rcx
  __int64 v9; // rdx
  unsigned __int64 v10; // rdx
  __int64 v12; // [rsp+0h] [rbp-50h] BYREF
  __int64 v13; // [rsp+8h] [rbp-48h]
  __int64 v14; // [rsp+10h] [rbp-40h]
  __int64 v15; // [rsp+18h] [rbp-38h]
  __int64 v16; // [rsp+20h] [rbp-30h]
  __int64 v17; // [rsp+28h] [rbp-28h]

  v3 = *(_WORD *)a1;
  if ( *(_WORD *)a1 == (_WORD)a2 )
  {
    LOBYTE(v4) = 0;
    if ( v3 || *(_QWORD *)(a1 + 8) == a3 )
      return v4;
    v12 = a2;
    v13 = a3;
  }
  else
  {
    v12 = a2;
    v13 = a3;
    if ( (_WORD)a2 )
    {
      if ( (_WORD)a2 == 1 || (unsigned __int16)(a2 - 504) <= 7u )
        goto LABEL_12;
      v5 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)a2 - 16];
      v7 = byte_444C4A0[16 * (unsigned __int16)a2 - 8];
      if ( !v3 )
        goto LABEL_6;
LABEL_14:
      if ( v3 != 1 && (unsigned __int16)(v3 - 504) > 7u )
      {
        v10 = *(_QWORD *)&byte_444C4A0[16 * v3 - 16];
        LOBYTE(v4) = byte_444C4A0[16 * v3 - 8];
        goto LABEL_7;
      }
LABEL_12:
      BUG();
    }
  }
  v16 = sub_3007260((__int64)&v12);
  v5 = v16;
  v17 = v6;
  v7 = v6;
  if ( v3 )
    goto LABEL_14;
LABEL_6:
  v8 = sub_3007260(a1);
  v4 = v9;
  v14 = v8;
  v10 = v8;
  v15 = v4;
LABEL_7:
  if ( (_BYTE)v4 || !v7 )
    LOBYTE(v4) = v5 < v10;
  return v4;
}
