// Function: sub_10A8400
// Address: 0x10a8400
//
__int64 __fastcall sub_10A8400(__int64 a1, int a2, unsigned __int8 *a3)
{
  __int64 result; // rax
  __int64 v5; // rax
  _BYTE *v6; // r13
  char v7; // al
  __int64 v8; // rax
  _BYTE *v9; // rbx
  char v10; // al
  _BYTE *v11; // r13
  unsigned __int64 v12; // rax
  char v13; // dl
  _BYTE *v14; // rbx
  unsigned __int64 v15; // rax
  char v16; // dl
  _BYTE *v17; // r14
  unsigned __int64 v18; // rax
  char v19; // dl
  _BYTE *v20; // r13
  unsigned __int64 v21; // rax
  char v22; // dl

  if ( a2 + 29 != *a3 )
    return 0;
  v5 = *((_QWORD *)a3 - 8);
  if ( !v5 )
    goto LABEL_7;
  **(_QWORD **)a1 = v5;
  v6 = (_BYTE *)*((_QWORD *)a3 - 4);
  if ( !v6 )
    return 0;
  **(_QWORD **)(a1 + 8) = v6;
  v7 = *v6;
  if ( *v6 != 68 )
    goto LABEL_6;
  v17 = (_BYTE *)*((_QWORD *)v6 - 4);
  if ( !v17 )
  {
LABEL_7:
    v8 = *((_QWORD *)a3 - 4);
    if ( !v8 )
      return 0;
    **(_QWORD **)a1 = v8;
    v9 = (_BYTE *)*((_QWORD *)a3 - 8);
    if ( !v9 )
      return 0;
    **(_QWORD **)(a1 + 8) = v9;
    v10 = *v9;
    if ( *v9 == 68 )
    {
      v20 = (_BYTE *)*((_QWORD *)v9 - 4);
      if ( !v20 )
        return 0;
      **(_QWORD **)(a1 + 16) = v20;
      if ( *v20 == 82 )
      {
        v21 = sub_B53900((__int64)v20);
        sub_B53630(v21, *(_QWORD *)(a1 + 24));
        if ( v22 )
        {
          if ( *((_QWORD *)v20 - 8) == **(_QWORD **)(a1 + 32) )
          {
            result = sub_F11D70((_QWORD **)(a1 + 40), *((_BYTE **)v20 - 4));
            if ( (_BYTE)result )
              return result;
          }
        }
      }
      v10 = *v9;
    }
    if ( v10 == 69 )
    {
      v14 = (_BYTE *)*((_QWORD *)v9 - 4);
      if ( v14 )
      {
        **(_QWORD **)(a1 + 56) = v14;
        if ( *v14 == 82 )
        {
          v15 = sub_B53900((__int64)v14);
          sub_B53630(v15, *(_QWORD *)(a1 + 64));
          if ( v16 )
          {
            if ( *((_QWORD *)v14 - 8) == **(_QWORD **)(a1 + 72) )
            {
              result = sub_F11D70((_QWORD **)(a1 + 80), *((_BYTE **)v14 - 4));
              if ( (_BYTE)result )
                return result;
            }
          }
        }
      }
    }
    return 0;
  }
  **(_QWORD **)(a1 + 16) = v17;
  if ( *v17 != 82
    || (v18 = sub_B53900((__int64)v17), sub_B53630(v18, *(_QWORD *)(a1 + 24)), !v19)
    || *((_QWORD *)v17 - 8) != **(_QWORD **)(a1 + 32)
    || (result = sub_F11D70((_QWORD **)(a1 + 40), *((_BYTE **)v17 - 4)), !(_BYTE)result) )
  {
    v7 = *v6;
LABEL_6:
    if ( v7 == 69 )
    {
      v11 = (_BYTE *)*((_QWORD *)v6 - 4);
      if ( v11 )
      {
        **(_QWORD **)(a1 + 56) = v11;
        if ( *v11 == 82 )
        {
          v12 = sub_B53900((__int64)v11);
          sub_B53630(v12, *(_QWORD *)(a1 + 64));
          if ( v13 )
          {
            if ( *((_QWORD *)v11 - 8) == **(_QWORD **)(a1 + 72) )
            {
              result = sub_F11D70((_QWORD **)(a1 + 80), *((_BYTE **)v11 - 4));
              if ( (_BYTE)result )
                return result;
            }
          }
        }
      }
    }
    goto LABEL_7;
  }
  return result;
}
