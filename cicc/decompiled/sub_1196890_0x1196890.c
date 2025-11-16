// Function: sub_1196890
// Address: 0x1196890
//
__int64 __fastcall sub_1196890(_QWORD **a1, _BYTE *a2, __int64 a3)
{
  char v3; // al
  char v4; // al
  _BYTE *v5; // r14
  _BYTE *v6; // rbx
  __int64 result; // rax
  _BYTE *v8; // r13
  __int64 v9; // r15
  unsigned int v10; // r14d
  bool v11; // al
  __int64 v12; // rdx
  __int64 v13; // r14
  _BYTE *v14; // rax
  unsigned int v15; // r14d
  __int64 v16; // rcx
  unsigned int v17; // r14d
  __int64 v18; // rax
  char v19; // [rsp+8h] [rbp-38h]
  int v20; // [rsp+8h] [rbp-38h]
  int v21; // [rsp+Ch] [rbp-34h]

  v3 = *a2;
  if ( *a2 != 42 )
  {
LABEL_2:
    if ( v3 == 59 )
    {
      v4 = sub_995B10(a1 + 3, *((_QWORD *)a2 - 8));
      v5 = (_BYTE *)*((_QWORD *)a2 - 4);
      if ( v4 && *v5 == 54 )
      {
        result = sub_995B10(a1 + 4, *((_QWORD *)v5 - 8));
        if ( (_BYTE)result )
        {
          v16 = *((_QWORD *)v5 - 4);
          if ( v16 )
            goto LABEL_28;
        }
        v5 = (_BYTE *)*((_QWORD *)a2 - 4);
      }
      if ( (unsigned __int8)sub_995B10(a1 + 3, (__int64)v5) )
      {
        v6 = (_BYTE *)*((_QWORD *)a2 - 8);
        if ( *v6 == 54 )
        {
          result = sub_995B10(a1 + 4, *((_QWORD *)v6 - 8));
          if ( (_BYTE)result )
          {
            v16 = *((_QWORD *)v6 - 4);
            if ( v16 )
            {
LABEL_28:
              *a1[5] = v16;
              return result;
            }
          }
        }
      }
    }
    return 0;
  }
  v8 = (_BYTE *)*((_QWORD *)a2 - 8);
  if ( *v8 != 54 )
    return 0;
  v9 = *((_QWORD *)v8 - 8);
  if ( *(_BYTE *)v9 == 17 )
  {
    v10 = *(_DWORD *)(v9 + 32);
    if ( v10 <= 0x40 )
      v11 = *(_QWORD *)(v9 + 24) == 1;
    else
      v11 = v10 - 1 == (unsigned int)sub_C444A0(v9 + 24);
LABEL_13:
    if ( v11 )
      goto LABEL_14;
LABEL_18:
    v3 = *a2;
    goto LABEL_2;
  }
  v13 = *(_QWORD *)(v9 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v13 + 8) - 17 > 1 || *(_BYTE *)v9 > 0x15u )
    return 0;
  v14 = sub_AD7630(*((_QWORD *)v8 - 8), 0, a3);
  if ( !v14 || *v14 != 17 )
  {
    if ( *(_BYTE *)(v13 + 8) == 17 )
    {
      v21 = *(_DWORD *)(v13 + 32);
      if ( v21 )
      {
        v19 = 0;
        v17 = 0;
        while ( 1 )
        {
          v18 = sub_AD69F0((unsigned __int8 *)v9, v17);
          if ( !v18 )
            break;
          if ( *(_BYTE *)v18 != 13 )
          {
            if ( *(_BYTE *)v18 != 17 )
              goto LABEL_18;
            if ( *(_DWORD *)(v18 + 32) <= 0x40u )
            {
              if ( *(_QWORD *)(v18 + 24) != 1 )
                goto LABEL_18;
            }
            else
            {
              v20 = *(_DWORD *)(v18 + 32);
              if ( (unsigned int)sub_C444A0(v18 + 24) != v20 - 1 )
                goto LABEL_18;
            }
            v19 = 1;
          }
          if ( v21 == ++v17 )
          {
            if ( v19 )
              goto LABEL_14;
            goto LABEL_18;
          }
        }
      }
    }
    goto LABEL_18;
  }
  v15 = *((_DWORD *)v14 + 8);
  if ( v15 > 0x40 )
  {
    v11 = v15 - 1 == (unsigned int)sub_C444A0((__int64)(v14 + 24));
    goto LABEL_13;
  }
  if ( *((_QWORD *)v14 + 3) != 1 )
    goto LABEL_18;
LABEL_14:
  if ( *a1 )
    **a1 = v9;
  v12 = *((_QWORD *)v8 - 4);
  if ( !v12 )
    goto LABEL_18;
  *a1[1] = v12;
  result = sub_995B10(a1 + 2, *((_QWORD *)a2 - 4));
  if ( !(_BYTE)result )
    goto LABEL_18;
  return result;
}
