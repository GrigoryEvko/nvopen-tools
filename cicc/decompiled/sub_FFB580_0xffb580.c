// Function: sub_FFB580
// Address: 0xffb580
//
__int64 __fastcall sub_FFB580(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // rsi
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // rax
  __int64 v6; // r15
  int v7; // r13d
  unsigned int v8; // ebx
  unsigned int v9; // r14d
  __int64 v10; // rax
  bool v11; // dl
  unsigned int v12; // r8d
  int v14; // eax
  __int64 v15; // r8
  __int64 v16; // [rsp+8h] [rbp-48h]
  int v17; // [rsp+14h] [rbp-3Ch]
  __int64 v18; // [rsp+18h] [rbp-38h]

  v3 = (_QWORD *)(a2 + 48);
  v4 = a3 & 0xFFFFFFFFFFFFFFF8LL;
  v16 = a3 >> 2;
  v18 = (a3 >> 2) & 1;
  v5 = *v3 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v5 == v3 )
    goto LABEL_19;
  if ( !v5 )
    BUG();
  v6 = v5 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v5 - 24) - 30 > 0xA )
    goto LABEL_19;
  v17 = sub_B46E30(v6);
  v7 = v17 >> 2;
  if ( v17 >> 2 > 0 )
  {
    v8 = 0;
    while ( 1 )
    {
      v10 = sub_B46EC0(v6, v8);
      if ( v4 == v10 )
        break;
      v9 = v8 + 1;
      v10 = sub_B46EC0(v6, v8 + 1);
      if ( v4 == v10
        || (v9 = v8 + 2, v10 = sub_B46EC0(v6, v8 + 2), v4 == v10)
        || (v9 = v8 + 3, v10 = sub_B46EC0(v6, v8 + 3), v4 == v10) )
      {
        v11 = v17 == v9;
        LOBYTE(v10) = v17 != v9;
        goto LABEL_15;
      }
      v8 += 4;
      if ( !--v7 )
      {
        v14 = v17 - v8;
        goto LABEL_21;
      }
    }
    v11 = v8 == v17;
    LOBYTE(v10) = v8 != v17;
    if ( !v18 )
      goto LABEL_16;
    goto LABEL_12;
  }
  v14 = v17;
  v8 = 0;
LABEL_21:
  if ( v14 == 2 )
    goto LABEL_22;
  if ( v14 != 3 )
  {
    if ( v14 == 1 )
      goto LABEL_24;
LABEL_19:
    LODWORD(v10) = 0;
    v11 = 1;
    goto LABEL_15;
  }
  v10 = sub_B46EC0(v6, v8);
  if ( v4 == v10 )
    goto LABEL_25;
  ++v8;
LABEL_22:
  v10 = sub_B46EC0(v6, v8);
  if ( v4 != v10 )
  {
    ++v8;
LABEL_24:
    v10 = sub_B46EC0(v6, v8);
    v11 = 1;
    v15 = v10;
    LODWORD(v10) = 0;
    if ( v4 != v15 )
      goto LABEL_15;
  }
LABEL_25:
  v11 = v17 == v8;
  LOBYTE(v10) = v17 != v8;
LABEL_15:
  if ( !v18 )
  {
LABEL_16:
    v12 = 0;
    if ( v11 )
      return v12;
  }
LABEL_12:
  LOBYTE(v10) = v16 & v10;
  return (unsigned int)v10 ^ 1;
}
