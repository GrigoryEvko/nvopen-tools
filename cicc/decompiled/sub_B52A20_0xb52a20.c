// Function: sub_B52A20
// Address: 0xb52a20
//
__int64 __fastcall sub_B52A20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rdi
  __int64 v7; // r12
  __int64 v9; // r13
  unsigned __int8 *v10; // rdx
  unsigned __int8 v11; // al
  __int64 v12; // rbx
  __int64 v13; // rdi
  char v14; // al
  __int64 v15; // r12
  __int64 v16; // rdi
  _BYTE *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  _BYTE *v21; // rbx
  _BYTE *v22; // r13
  _BYTE *v23; // rdi
  int v25; // r15d
  unsigned int v26; // r14d
  _BYTE *v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  _BYTE *v31; // rbx
  char v32; // al
  __int64 v33; // r12
  _BYTE *v34; // rdi
  char v35; // al
  _BYTE *v36; // rdx

  v6 = *(_WORD *)(a1 + 2) & 0x3F;
  if ( (_BYTE)a2 )
    v6 = (unsigned int)sub_B52870(v6);
  switch ( (_DWORD)v6 )
  {
    case 9:
      v6 = a1;
      LODWORD(v7) = sub_B451C0(a1);
      if ( !(_BYTE)v7 )
        return (unsigned int)v7;
      break;
    case 0x20:
      LODWORD(v7) = 1;
      return (unsigned int)v7;
    case 1:
      break;
    default:
      goto LABEL_6;
  }
  v9 = *(_QWORD *)(a1 - 64);
  v10 = *(unsigned __int8 **)(a1 - 32);
  v11 = *(_BYTE *)v9;
  if ( *(_BYTE *)v9 <= 0x15u )
  {
    if ( v10 )
      goto LABEL_11;
LABEL_31:
    BUG();
  }
  if ( !v10 )
    goto LABEL_31;
  v11 = *v10;
  LODWORD(v7) = 0;
  v9 = *(_QWORD *)(a1 - 32);
  if ( *v10 > 0x15u )
    return (unsigned int)v7;
LABEL_11:
  if ( v11 == 18 )
  {
    v7 = v9 + 24;
    v12 = sub_C33340(v6, a2, v10, a4, a5);
    v13 = v9 + 24;
    if ( *(_QWORD *)(v9 + 24) == v12 )
      v14 = sub_C40310(v13);
    else
      v14 = sub_C33940(v13);
    if ( !v14 )
    {
      if ( v12 == *(_QWORD *)(v9 + 24) )
        v7 = *(_QWORD *)(v9 + 32);
      LOBYTE(v7) = (*(_BYTE *)(v7 + 20) & 7) != 3;
      return (unsigned int)v7;
    }
    goto LABEL_6;
  }
  v15 = *(_QWORD *)(v9 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v15 + 8) - 17 > 1 )
    goto LABEL_6;
  v16 = v9;
  v17 = sub_AD7630(v9, 0, (__int64)v10);
  v21 = v17;
  if ( !v17 || *v17 != 18 )
  {
    if ( *(_BYTE *)(v15 + 8) == 17 )
    {
      v25 = *(_DWORD *)(v15 + 32);
      if ( v25 )
      {
        LODWORD(v7) = 0;
        v26 = 0;
        while ( 1 )
        {
          v27 = (_BYTE *)sub_AD69F0((unsigned __int8 *)v9, v26);
          v31 = v27;
          if ( !v27 )
            break;
          v32 = *v27;
          if ( v32 != 13 )
          {
            if ( v32 != 18 )
              goto LABEL_6;
            v33 = sub_C33340(v9, v26, v28, v29, v30);
            v34 = v31 + 24;
            v35 = *((_QWORD *)v31 + 3) == v33 ? sub_C40310(v34) : sub_C33940(v34);
            v36 = v31 + 24;
            if ( v35 )
              goto LABEL_6;
            if ( v33 == *((_QWORD *)v31 + 3) )
              v36 = (_BYTE *)*((_QWORD *)v31 + 4);
            if ( (v36[20] & 7) == 3 )
              goto LABEL_6;
            LODWORD(v7) = 1;
          }
          if ( v25 == ++v26 )
            return (unsigned int)v7;
        }
      }
    }
    goto LABEL_6;
  }
  v22 = v17 + 24;
  v7 = sub_C33340(v16, 0, v18, v19, v20);
  v23 = v21 + 24;
  if ( *((_QWORD *)v21 + 3) == v7 ? sub_C40310(v23) : (unsigned __int8)sub_C33940(v23) )
  {
LABEL_6:
    LODWORD(v7) = 0;
    return (unsigned int)v7;
  }
  if ( v7 == *((_QWORD *)v21 + 3) )
    v22 = (_BYTE *)*((_QWORD *)v21 + 4);
  LOBYTE(v7) = (v22[20] & 7) != 3;
  return (unsigned int)v7;
}
