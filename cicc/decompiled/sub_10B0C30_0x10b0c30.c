// Function: sub_10B0C30
// Address: 0x10b0c30
//
bool __fastcall sub_10B0C30(_QWORD *a1, int a2, unsigned __int8 *a3)
{
  _BYTE *v4; // rax
  _BYTE *v5; // r13
  __int64 v6; // rax
  __int64 v7; // r14
  unsigned int v8; // r15d
  __int64 v9; // rax
  _BYTE *v10; // rax
  int v11; // eax
  unsigned __int8 *v12; // [rsp-30h] [rbp-30h]
  unsigned __int8 *v13; // [rsp-30h] [rbp-30h]
  unsigned __int8 *v14; // [rsp-30h] [rbp-30h]

  if ( a2 + 29 != *a3 )
    return 0;
  v4 = (_BYTE *)*((_QWORD *)a3 - 8);
  if ( *v4 != 55 || *((_QWORD *)v4 - 8) != *a1 )
    goto LABEL_4;
  v7 = *((_QWORD *)v4 - 4);
  if ( !v7 )
    BUG();
  if ( *(_BYTE *)v7 == 17
    || (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v7 + 8) + 8LL) - 17 <= 1
    && *(_BYTE *)v7 <= 0x15u
    && (v13 = a3, v10 = sub_AD7630(v7, 0, (__int64)a3), a3 = v13, (v7 = (__int64)v10) != 0)
    && *v10 == 17 )
  {
    v8 = *(_DWORD *)(v7 + 32);
    v5 = (_BYTE *)*((_QWORD *)a3 - 4);
    if ( v8 > 0x40 )
    {
      v14 = a3;
      v11 = sub_C444A0(v7 + 24);
      a3 = v14;
      if ( v8 - v11 > 0x40 )
        goto LABEL_5;
      v9 = **(_QWORD **)(v7 + 24);
    }
    else
    {
      v9 = *(_QWORD *)(v7 + 24);
    }
    if ( a1[1] == v9 && ((_BYTE *)a1[2] == v5 || (_BYTE *)a1[3] == v5) )
      return 1;
  }
  else
  {
LABEL_4:
    v5 = (_BYTE *)*((_QWORD *)a3 - 4);
  }
LABEL_5:
  if ( *v5 == 55 && *((_QWORD *)v5 - 8) == *a1 )
  {
    v12 = a3;
    if ( sub_F17ED0(a1 + 1, *((_QWORD *)v5 - 4)) )
    {
      v6 = *((_QWORD *)v12 - 8);
      if ( v6 != a1[2] )
        return a1[3] == v6;
      return 1;
    }
  }
  return 0;
}
