// Function: sub_9858D0
// Address: 0x9858d0
//
__int64 __fastcall sub_9858D0(__int64 a1, char a2, _BYTE *a3, char a4)
{
  __int64 v4; // rax
  _BYTE *v5; // r12
  __int64 v7; // rdx
  __int64 v9; // rdi
  _QWORD **v10; // r13
  __int64 v11; // rdi
  unsigned int v12; // ebx
  _QWORD **v13; // rax
  unsigned int v14; // ebx
  _QWORD *v15; // rax
  _BYTE *v16; // rax

  if ( *a3 != 82 )
    goto LABEL_3;
  v4 = *((_QWORD *)a3 - 8);
  v5 = a3;
  if ( *(_BYTE *)v4 != 85 )
    goto LABEL_3;
  v7 = *(_QWORD *)(v4 - 32);
  if ( !v7
    || *(_BYTE *)v7
    || *(_QWORD *)(v7 + 24) != *(_QWORD *)(v4 + 80)
    || *(_DWORD *)(v7 + 36) != 66
    || a1 != *(_QWORD *)(v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF)) )
  {
    goto LABEL_3;
  }
  v9 = *((_QWORD *)v5 - 4);
  v10 = (_QWORD **)(v9 + 24);
  if ( *(_BYTE *)v9 != 17 )
  {
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v9 + 8) + 8LL) - 17 > 1 )
      goto LABEL_3;
    if ( *(_BYTE *)v9 > 0x15u )
      goto LABEL_3;
    v16 = (_BYTE *)sub_AD7630(v9, 0);
    if ( !v16 || *v16 != 17 )
      goto LABEL_3;
    v10 = (_QWORD **)(v16 + 24);
  }
  v11 = sub_B53900(v5) & 0xFFFFFFFFFFLL;
  if ( !a4 )
    LODWORD(v11) = sub_B52870(v11);
  if ( a2 && (_DWORD)v11 == 36 )
  {
    v14 = *((_DWORD *)v10 + 2);
    if ( v14 <= 0x40 )
    {
      v15 = *v10;
LABEL_21:
      LOBYTE(v5) = v15 == (_QWORD *)2;
      return (unsigned int)v5;
    }
    if ( v14 - (unsigned int)sub_C444A0(v10) <= 0x40 )
    {
      v15 = (_QWORD *)**v10;
      goto LABEL_21;
    }
LABEL_3:
    LODWORD(v5) = 0;
    return (unsigned int)v5;
  }
  LODWORD(v5) = 0;
  if ( (_DWORD)v11 != 32 )
    return (unsigned int)v5;
  v12 = *((_DWORD *)v10 + 2);
  if ( v12 > 0x40 )
  {
    if ( v12 - (unsigned int)sub_C444A0(v10) <= 0x40 )
    {
      v13 = (_QWORD **)**v10;
      goto LABEL_18;
    }
    return (unsigned int)v5;
  }
  v13 = (_QWORD **)*v10;
LABEL_18:
  LOBYTE(v5) = v13 == (_QWORD **)1;
  return (unsigned int)v5;
}
