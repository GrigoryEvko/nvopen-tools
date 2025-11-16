// Function: sub_656C00
// Address: 0x656c00
//
__int64 __fastcall sub_656C00(__int64 a1, __int64 a2, __int64 a3, int a4, int a5)
{
  __int64 v8; // rdi
  _DWORD *v9; // r14
  int v10; // r12d
  int v11; // eax
  __int64 v12; // rdx
  __int64 v13; // r8
  __int64 result; // rax
  __int64 v15; // rsi
  int v16; // r10d
  char v17; // di
  __int64 v18; // rdi
  char v19; // cl
  __int64 v20; // rdi
  unsigned int v21; // eax
  __int64 v22; // rdx
  unsigned int v23; // eax
  __int64 v24; // rdi
  __int64 v25; // rdx
  unsigned int v26; // [rsp+8h] [rbp-128h]
  unsigned int v28; // [rsp+18h] [rbp-118h]
  __int64 v29; // [rsp+18h] [rbp-118h]
  char s[112]; // [rsp+20h] [rbp-110h] BYREF
  char v31[160]; // [rsp+90h] [rbp-A0h] BYREF

  switch ( (_BYTE)a2 )
  {
    case 7:
      v18 = *(_QWORD *)(a3 + 120);
      v9 = (_DWORD *)(a3 + 152);
      if ( *(char *)(v18 + 142) >= 0 && *(_BYTE *)(v18 + 140) == 12 )
        v28 = sub_8D4AB0(v18, a2, a3);
      else
        v28 = *(_DWORD *)(v18 + 136);
      v10 = *(_DWORD *)(a3 + 152) != 0;
      v13 = 0;
      break;
    case 6:
      if ( !(unsigned int)sub_8D3A70(a3) && !(unsigned int)sub_8D2870(a3) )
      {
        result = *(unsigned int *)(a1 + 376);
        if ( (_DWORD)result )
          goto LABEL_41;
        return result;
      }
      v9 = (_DWORD *)(a3 + 136);
      if ( *(char *)(a3 + 142) >= 0 && *(_BYTE *)(a3 + 140) == 12 )
        v28 = sub_8D4AB0(a3, a2, v25);
      else
        v28 = *(_DWORD *)(a3 + 136);
      v13 = a3;
      v10 = *(_BYTE *)(a3 + 142) >> 7;
      break;
    case 8:
      v8 = *(_QWORD *)(a3 + 120);
      v9 = (_DWORD *)(a3 + 140);
      if ( *(char *)(v8 + 142) >= 0 && *(_BYTE *)(v8 + 140) == 12 )
        v28 = sub_8D4AB0(v8, a2, a3);
      else
        v28 = *(_DWORD *)(v8 + 136);
      v10 = *(_DWORD *)(a3 + 140) != 0;
      v11 = sub_7A7D20(v8, a2, a3);
      v13 = 0;
      if ( v11 )
      {
        v26 = *(_DWORD *)(a1 + 376);
        v21 = sub_7A7D20(v8, a2, v12);
        v13 = 0;
        if ( v26 > v21 )
        {
          v23 = sub_7A7D20(v8, a2, v22);
          v13 = 0;
          v28 = v23;
        }
      }
      break;
    default:
      sub_721090(a1);
  }
  result = *(unsigned int *)(a1 + 376);
  if ( (_DWORD)result )
  {
    v15 = *(_QWORD *)(a1 + 384);
    v16 = 1;
    if ( v15 )
    {
      v17 = *(_BYTE *)(v15 + 9);
      if ( v17 == 1 || (v16 = 0, v17 == 4) )
      {
        v16 = 0;
        if ( (*(_BYTE *)(v15 + 11) & 0x10) == 0 )
        {
          v16 = 1;
          if ( !v9 )
            goto LABEL_41;
        }
      }
    }
    if ( v28 <= (unsigned int)result )
    {
      if ( !v10 )
        goto LABEL_28;
    }
    else if ( !v10 )
    {
      if ( !HIDWORD(qword_4F077B4) )
      {
LABEL_55:
        result = sub_6851C0(1881, v15 + 56);
        goto LABEL_41;
      }
      if ( v16 )
      {
        if ( dword_4F077C4 == 2 && !(_DWORD)qword_4F077B4 )
          goto LABEL_18;
        goto LABEL_55;
      }
LABEL_28:
      if ( a4 )
      {
        if ( (_DWORD)qword_4F077B4 )
        {
          v29 = v13;
          sub_6854F0(5, 1887, v15 + 56, a3 + 64);
          result = *(unsigned int *)(a1 + 376);
          v13 = v29;
        }
        else if ( !HIDWORD(qword_4F077B4) )
        {
          result = sub_6854F0(8, 1887, v15 + 56, a3 + 64);
LABEL_41:
          *(_DWORD *)(a1 + 376) = 0;
          return result;
        }
      }
LABEL_18:
      *v9 = result;
      if ( v13 )
        *(_BYTE *)(v13 + 142) |= 0x80u;
      goto LABEL_41;
    }
    if ( *v9 != (_DWORD)result )
    {
      sprintf(s, "%d", *v9);
      sprintf(v31, "%d", *(_DWORD *)(a1 + 376));
      v24 = 8;
      if ( !a5 )
      {
        v24 = 7;
        if ( HIDWORD(qword_4F077B4) )
          v24 = (_DWORD)qword_4F077B4 == 0 ? 5 : 7;
      }
      result = sub_6864B0(v24, 1886, *(_QWORD *)(*(_QWORD *)(a1 + 384) + 40LL), v31, s);
    }
    goto LABEL_41;
  }
  if ( a5 )
  {
    if ( v10 )
    {
      result = sub_736C60(3, *(_QWORD *)(a3 + 104));
      if ( result )
      {
        v19 = *(_BYTE *)(result + 9);
        if ( (v19 == 1 || v19 == 4) && (*(_BYTE *)(result + 11) & 0x10) == 0 )
        {
          v20 = 8;
          if ( HIDWORD(qword_4F077B4) )
            v20 = (_DWORD)qword_4F077B4 == 0 ? 5 : 8;
          return sub_6854F0(v20, 1887, result + 56, a3 + 64);
        }
      }
    }
  }
  return result;
}
