// Function: sub_FEF800
// Address: 0xfef800
//
__int64 __fastcall sub_FEF800(__int64 a1, __int64 a2)
{
  int v2; // r14d
  _QWORD *v3; // r12
  _QWORD *v4; // rbx
  int v5; // edx
  int v6; // eax
  __int64 v8; // rax
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  _QWORD *v11; // rbx
  char v12; // al
  __int64 v13; // [rsp+8h] [rbp-28h]

  v3 = (_QWORD *)(a2 + 48);
  v4 = (_QWORD *)(*(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (_QWORD *)(a2 + 48) == v4 )
    goto LABEL_16;
  if ( !v4 )
    goto LABEL_30;
  v5 = *((unsigned __int8 *)v4 - 24);
  if ( (unsigned int)(v5 - 30) > 0xA )
LABEL_16:
    BUG();
  if ( (_BYTE)v5 == 36 )
    goto LABEL_7;
  if ( sub_AA4F10(a2) )
  {
    v4 = (_QWORD *)(*(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL);
    if ( v3 == v4 )
    {
LABEL_15:
      v6 = 0;
    }
    else
    {
      while ( 1 )
      {
LABEL_7:
        if ( !v4 )
          goto LABEL_30;
        if ( *((_BYTE *)v4 - 24) == 85
          && ((unsigned __int8)sub_A73ED0(v4 + 6, 36) || (unsigned __int8)sub_B49560((__int64)(v4 - 3), 36)) )
        {
          break;
        }
        v4 = (_QWORD *)(*v4 & 0xFFFFFFFFFFFFFFF8LL);
        if ( v3 == v4 )
          goto LABEL_15;
      }
      v6 = 1;
    }
    LODWORD(v13) = v6;
    BYTE4(v13) = 1;
    return v13;
  }
  v8 = sub_AA4FF0(a2);
  if ( !v8 )
LABEL_30:
    BUG();
  v9 = (unsigned int)*(unsigned __int8 *)(v8 - 24) - 39;
  if ( (unsigned int)v9 <= 0x38 && (v10 = 0x100060000000001LL, _bittest64(&v10, v9)) )
  {
    v12 = 1;
    v2 = 1;
  }
  else
  {
    v11 = *(_QWORD **)(a2 + 56);
    if ( v3 == v11 )
    {
LABEL_28:
      v12 = 0;
    }
    else
    {
      while ( 1 )
      {
        if ( !v11 )
          goto LABEL_30;
        if ( *((_BYTE *)v11 - 24) == 85
          && ((unsigned __int8)sub_A73ED0(v11 + 6, 5) || (unsigned __int8)sub_B49560((__int64)(v11 - 3), 5)) )
        {
          break;
        }
        v11 = (_QWORD *)v11[1];
        if ( v3 == v11 )
          goto LABEL_28;
      }
      v12 = 1;
      v2 = 0xFFFF;
    }
  }
  LODWORD(v13) = v2;
  BYTE4(v13) = v12;
  return v13;
}
