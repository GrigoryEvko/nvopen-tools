// Function: sub_2AC3650
// Address: 0x2ac3650
//
char __fastcall sub_2AC3650(__int64 a1, unsigned __int8 *a2, unsigned __int64 a3)
{
  unsigned int v3; // r13d
  unsigned __int64 v5; // rbx
  __int64 v6; // rax
  int v7; // ecx
  unsigned int v8; // edx
  unsigned __int8 v9; // r15
  bool v10; // sf
  bool v11; // of
  __int64 *v12; // r15
  __int64 v13; // rdi
  char v14; // r8
  char v16; // [rsp+0h] [rbp-60h]
  __int64 v17; // [rsp+0h] [rbp-60h]
  unsigned __int8 *v19; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v20; // [rsp+18h] [rbp-48h]
  char v21; // [rsp+1Ch] [rbp-44h]
  __int64 v22; // [rsp+20h] [rbp-40h]
  int v23; // [rsp+28h] [rbp-38h]

  v3 = a3;
  v5 = HIDWORD(a3);
  LOBYTE(v6) = sub_2AB37C0(a1, a2);
  if ( !(_BYTE)v6 )
    return v6;
  v7 = *a2;
  v8 = v7 - 29;
  v9 = *a2;
  if ( (unsigned int)(v7 - 29) > 0x21 )
  {
    if ( v7 == 85 )
    {
      LOBYTE(v6) = (v5 ^ 1) & (v3 == 1);
      if ( !(_BYTE)v6 )
      {
        v19 = a2;
        v20 = v3;
        v21 = v5;
        v6 = sub_2AC3590(a1 + 384, (__int64 *)&v19);
        if ( !v6 )
          v6 = *(_QWORD *)(a1 + 392) + ((unsigned __int64)*(unsigned int *)(a1 + 408) << 6);
        LOBYTE(v6) = *(_DWORD *)(v6 + 16) == 5;
      }
    }
    return v6;
  }
  if ( v8 <= 0x1F )
  {
    if ( v8 > 0x14 )
    {
      if ( (unsigned int)(v7 - 51) > 1 )
        return v6;
    }
    else if ( v8 <= 0x12 )
    {
      return v6;
    }
    v16 = v6;
    sub_2AC1010((__int64)&v19, a1, a2, a3);
    if ( (_DWORD)qword_500CF68 != 1 )
    {
      LOBYTE(v6) = v16;
      if ( (_DWORD)qword_500CF68 != 2 )
      {
        if ( (_DWORD)qword_500CF68 )
          BUG();
        v11 = __OFSUB__(v20, v23);
        v10 = (int)(v20 - v23) < 0;
        if ( v20 == v23 )
        {
          v11 = __OFSUB__(v19, v22);
          v10 = (__int64)&v19[-v22] < 0;
        }
        LOBYTE(v6) = v10 ^ v11;
      }
      return v6;
    }
LABEL_31:
    LOBYTE(v6) = 0;
    return v6;
  }
  v17 = sub_228AED0(a2);
  if ( v9 == 61 )
    v12 = (__int64 *)*((_QWORD *)a2 + 1);
  else
    v12 = *(__int64 **)(*((_QWORD *)a2 - 8) + 8LL);
  if ( (_BYTE)v5 )
  {
    if ( !v3 )
      goto LABEL_24;
  }
  else if ( v3 <= 1 )
  {
    goto LABEL_24;
  }
  sub_BCE1B0(v12, a3);
LABEL_24:
  sub_2AAE0E0((__int64)a2);
  v13 = *(_QWORD *)(a1 + 440);
  if ( *a2 == 61 )
  {
    if ( !(unsigned int)sub_31A5150(v13, v12, v17) || !(unsigned __int8)sub_DFA390(*(_QWORD *)(a1 + 448)) )
    {
      LODWORD(v6) = sub_DFA510(*(_QWORD *)(a1 + 448)) ^ 1;
      return v6;
    }
    goto LABEL_31;
  }
  if ( !(unsigned int)sub_31A5150(v13, v12, v17) || (v14 = sub_DFA360(*(_QWORD *)(a1 + 448)), LOBYTE(v6) = 0, !v14) )
    LODWORD(v6) = sub_DFA570(*(_QWORD *)(a1 + 448)) ^ 1;
  return v6;
}
