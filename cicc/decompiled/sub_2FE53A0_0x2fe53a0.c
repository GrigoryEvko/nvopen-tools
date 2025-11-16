// Function: sub_2FE53A0
// Address: 0x2fe53a0
//
__int64 __fastcall sub_2FE53A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int16 v4; // r12
  unsigned __int16 v5; // bx
  unsigned __int64 v7; // r13
  __int64 v8; // rdx
  char v9; // r14
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  _QWORD v14[2]; // [rsp+0h] [rbp-60h] BYREF
  unsigned __int16 v15; // [rsp+10h] [rbp-50h] BYREF
  __int64 v16; // [rsp+18h] [rbp-48h]
  __int64 v17; // [rsp+20h] [rbp-40h]
  __int64 v18; // [rsp+28h] [rbp-38h]
  __int64 v19; // [rsp+30h] [rbp-30h]
  __int64 v20; // [rsp+38h] [rbp-28h]

  v4 = a3;
  v5 = *(_WORD *)(a1 + 2866);
  v14[0] = a3;
  v14[1] = a4;
  if ( v5 == (_WORD)a3 )
  {
    if ( v5 || !a4 )
      return v14[0];
    v16 = 0;
    v15 = 0;
LABEL_5:
    v19 = sub_3007260(&v15);
    v7 = v19;
    v20 = v8;
    v9 = v8;
    if ( !v4 )
    {
LABEL_6:
      v10 = sub_3007260(v14);
      v12 = v11;
      v17 = v10;
      v13 = v10;
      v18 = v12;
      goto LABEL_7;
    }
    goto LABEL_15;
  }
  v15 = v5;
  v16 = 0;
  if ( !v5 )
    goto LABEL_5;
  if ( v5 == 1 || (unsigned __int16)(v5 - 504) <= 7u )
LABEL_13:
    BUG();
  v7 = *(_QWORD *)&byte_444C4A0[16 * v5 - 16];
  v9 = byte_444C4A0[16 * v5 - 8];
  if ( !(_WORD)a3 )
    goto LABEL_6;
LABEL_15:
  if ( v4 == 1 || (unsigned __int16)(v4 - 504) <= 7u )
    goto LABEL_13;
  v13 = *(_QWORD *)&byte_444C4A0[16 * v4 - 16];
  LOBYTE(v12) = byte_444C4A0[16 * v4 - 8];
LABEL_7:
  if ( (_BYTE)v12 && !v9 || v13 >= v7 )
    return v14[0];
  return v5;
}
