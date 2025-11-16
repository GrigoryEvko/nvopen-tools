// Function: sub_15F3C00
// Address: 0x15f3c00
//
__int64 __fastcall sub_15F3C00(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rbx
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r14
  _QWORD *v9; // rbx
  __int64 result; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  _QWORD *v13; // r13
  _QWORD *v14; // rax
  __int64 v15; // rax

  if ( *(char *)(a1 + 23) >= 0 )
  {
    LODWORD(v5) = 0;
    if ( *(char *)(a2 + 23) >= 0 )
    {
      v9 = 0;
      goto LABEL_9;
    }
  }
  else
  {
    v2 = sub_1648A40(a1);
    v4 = v2 + v3;
    if ( *(char *)(a1 + 23) < 0 )
      v4 -= sub_1648A40(a1);
    v5 = v4 >> 4;
    if ( *(char *)(a2 + 23) >= 0 )
    {
      LODWORD(v15) = 0;
      goto LABEL_17;
    }
  }
  v6 = sub_1648A40(a2);
  v8 = v6 + v7;
  if ( *(char *)(a2 + 23) >= 0 )
  {
    v15 = v8 >> 4;
LABEL_17:
    if ( (_DWORD)v5 != (_DWORD)v15 )
      return 0;
LABEL_18:
    v9 = 0;
    goto LABEL_9;
  }
  if ( (_DWORD)v5 != (unsigned int)((v8 - sub_1648A40(a2)) >> 4) )
    return 0;
  if ( *(char *)(a2 + 23) >= 0 )
    goto LABEL_18;
  v9 = (_QWORD *)sub_1648A40(a2);
LABEL_9:
  result = 1;
  if ( *(char *)(a1 + 23) >= 0 )
    return result;
  v11 = sub_1648A40(a1);
  v13 = (_QWORD *)(v11 + v12);
  if ( *(char *)(a1 + 23) >= 0 )
    v14 = 0;
  else
    v14 = (_QWORD *)sub_1648A40(a1);
  if ( v14 == v13 )
    return 1;
  while ( *v14 == *v9 && v14[1] == v9[1] )
  {
    v14 += 2;
    v9 += 2;
    if ( v13 == v14 )
      return 1;
  }
  return 0;
}
