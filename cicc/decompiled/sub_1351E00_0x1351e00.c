// Function: sub_1351E00
// Address: 0x1351e00
//
_BYTE *__fastcall sub_1351E00(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int8 v4; // r12
  __int64 v5; // rax
  _WORD *v6; // rdx
  __int64 v7; // rdi
  __int64 v8; // rax
  _WORD *v9; // rdx
  __int64 v10; // r12
  __int64 v11; // rdx
  _BYTE *result; // rax

  v4 = a1;
  v5 = sub_16E8CB0(a1, a2, a3);
  v6 = *(_WORD **)(v5 + 24);
  v7 = v5;
  if ( *(_QWORD *)(v5 + 16) - (_QWORD)v6 <= 1u )
  {
    v7 = sub_16E7EE0(v5, "  ", 2);
  }
  else
  {
    *v6 = 8224;
    *(_QWORD *)(v5 + 24) += 2LL;
  }
  v8 = sub_134CED0(v7, v4);
  v9 = *(_WORD **)(v8 + 24);
  v10 = v8;
  if ( *(_QWORD *)(v8 + 16) - (_QWORD)v9 <= 1u )
  {
    v10 = sub_16E7EE0(v8, ": ", 2);
  }
  else
  {
    *v9 = 8250;
    *(_QWORD *)(v8 + 24) += 2LL;
  }
  sub_155C2B0(a2, v10, 0);
  v11 = *(_QWORD *)(v10 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(v10 + 16) - v11) <= 4 )
  {
    v10 = sub_16E7EE0(v10, " <-> ", 5);
  }
  else
  {
    *(_DWORD *)v11 = 1043151904;
    *(_BYTE *)(v11 + 4) = 32;
    *(_QWORD *)(v10 + 24) += 5LL;
  }
  sub_155C2B0(a3, v10, 0);
  result = *(_BYTE **)(v10 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(v10 + 16) )
    return (_BYTE *)sub_16E7DE0(v10, 10);
  *(_QWORD *)(v10 + 24) = result + 1;
  *result = 10;
  return result;
}
