// Function: sub_1550E20
// Address: 0x1550e20
//
_BYTE *__fastcall sub_1550E20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int8 v5; // al
  char v6; // r12
  int v7; // ebx
  _BYTE *result; // rax
  _BYTE *v9; // rax
  unsigned __int8 v10; // si
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r14
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r14

  if ( (*(_BYTE *)(a2 + 23) & 0x20) != 0 )
    return sub_154B790(a1, a2);
  v5 = *(_BYTE *)(a2 + 16);
  if ( v5 <= 0x10u )
  {
    if ( v5 > 3u )
      return (_BYTE *)sub_15510D0();
    if ( a4 )
    {
      v6 = 64;
      v7 = sub_154F210(a4, a2, a3, a4);
      goto LABEL_6;
    }
    goto LABEL_27;
  }
  if ( v5 != 20 )
  {
    if ( v5 == 19 )
      return sub_154F770(a1, *(unsigned __int8 **)(a2 + 24), a3, a4, a5);
    if ( a4 )
    {
      v7 = sub_154F3B0(a4, a2, a3, a4);
      if ( v7 != -1 )
      {
        v10 = 37;
LABEL_26:
        v11 = sub_1549FC0(a1, v10);
        return (_BYTE *)sub_16E7AB0(v11, v7);
      }
      v16 = sub_154BED0(a2);
      v19 = v16;
      if ( !v16 )
        return (_BYTE *)sub_1263B40(a1, "<badref>");
      v6 = 37;
      v7 = sub_154F3B0(v16, a2, v17, v18);
      sub_154C1A0(v19);
      j_j___libc_free_0(v19, 272);
LABEL_6:
      if ( v7 == -1 )
        return (_BYTE *)sub_1263B40(a1, "<badref>");
      v10 = v6;
      goto LABEL_26;
    }
LABEL_27:
    v12 = sub_154BED0(a2);
    v15 = v12;
    if ( !v12 )
      return (_BYTE *)sub_1263B40(a1, "<badref>");
    if ( *(_BYTE *)(a2 + 16) > 3u )
    {
      v6 = 37;
      v7 = sub_154F3B0(v12, a2, v13, v14);
    }
    else
    {
      v6 = 64;
      v7 = sub_154F210(v12, a2, v13, v14);
    }
    sub_154C1A0(v15);
    j_j___libc_free_0(v15, 272);
    goto LABEL_6;
  }
  sub_1263B40(a1, "asm ");
  if ( *(_BYTE *)(a2 + 96) )
    sub_1263B40(a1, "sideeffect ");
  if ( *(_BYTE *)(a2 + 97) )
    sub_1263B40(a1, "alignstack ");
  if ( *(_DWORD *)(a2 + 100) == 1 )
    sub_1263B40(a1, "inteldialect ");
  v9 = *(_BYTE **)(a1 + 24);
  if ( (unsigned __int64)v9 >= *(_QWORD *)(a1 + 16) )
  {
    sub_16E7DE0(a1, 34);
  }
  else
  {
    *(_QWORD *)(a1 + 24) = v9 + 1;
    *v9 = 34;
  }
  sub_16D16F0(*(_QWORD *)(a2 + 24), *(_QWORD *)(a2 + 32), a1);
  sub_1263B40(a1, "\", \"");
  sub_16D16F0(*(_QWORD *)(a2 + 56), *(_QWORD *)(a2 + 64), a1);
  result = *(_BYTE **)(a1 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(a1 + 16) )
    return (_BYTE *)sub_16E7DE0(a1, 34);
  *(_QWORD *)(a1 + 24) = result + 1;
  *result = 34;
  return result;
}
