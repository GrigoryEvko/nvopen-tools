// Function: sub_98ED80
// Address: 0x98ed80
//
__int64 __fastcall sub_98ED80(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // rcx
  unsigned int v5; // r8d
  unsigned int v6; // r13d
  __int64 v8; // r15
  char *v9; // rbx
  __int64 v10; // r15
  char *v11; // r15
  signed __int64 v12; // rax
  unsigned int v13; // r14d
  char *v14; // [rsp+8h] [rbp-38h]

  if ( sub_98ED70((unsigned __int8 *)a1, 0, 0, 0, 0) )
    return 1;
  v6 = sub_98D200((_BYTE *)a1, a2, 0, v4, v5);
  if ( (_BYTE)v6 )
    return 1;
  if ( a3 <= 1 && *(_BYTE *)a1 > 0x1Cu && !sub_98CD70((unsigned __int8 *)a1, 1) )
  {
    v8 = 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
    if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    {
      v9 = *(char **)(a1 - 8);
      v14 = &v9[v8];
    }
    else
    {
      v14 = (char *)a1;
      v9 = (char *)(a1 - v8);
    }
    v10 = v8 >> 7;
    if ( v10 )
    {
      v6 = a3 + 1;
      v11 = &v9[128 * v10];
      while ( (unsigned __int8)sub_98ED80(*(_QWORD *)v9, a2, v6) )
      {
        if ( !(unsigned __int8)sub_98ED80(*((_QWORD *)v9 + 4), a2, v6) )
        {
          LOBYTE(v6) = v14 == v9 + 32;
          return v6;
        }
        if ( !(unsigned __int8)sub_98ED80(*((_QWORD *)v9 + 8), a2, v6) )
        {
          LOBYTE(v6) = v14 == v9 + 64;
          return v6;
        }
        if ( !(unsigned __int8)sub_98ED80(*((_QWORD *)v9 + 12), a2, v6) )
        {
          LOBYTE(v6) = v14 == v9 + 96;
          return v6;
        }
        v9 += 128;
        if ( v11 == v9 )
          goto LABEL_19;
      }
      goto LABEL_17;
    }
LABEL_19:
    v12 = v14 - v9;
    if ( v14 - v9 == 64 )
    {
      v13 = a3 + 1;
    }
    else
    {
      if ( v12 != 96 )
      {
        if ( v12 == 32 )
        {
          v13 = a3 + 1;
          goto LABEL_23;
        }
        return 1;
      }
      v13 = a3 + 1;
      if ( !(unsigned __int8)sub_98ED80(*(_QWORD *)v9, a2, v13) )
      {
LABEL_17:
        LOBYTE(v6) = v14 == v9;
        return v6;
      }
      v9 += 32;
    }
    if ( (unsigned __int8)sub_98ED80(*(_QWORD *)v9, a2, v13) )
    {
      v9 += 32;
LABEL_23:
      v6 = sub_98ED80(*(_QWORD *)v9, a2, v13);
      if ( (_BYTE)v6 )
        return v6;
      goto LABEL_17;
    }
    goto LABEL_17;
  }
  return v6;
}
