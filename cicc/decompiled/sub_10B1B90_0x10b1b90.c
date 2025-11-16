// Function: sub_10B1B90
// Address: 0x10b1b90
//
bool __fastcall sub_10B1B90(__int64 a1, int a2, unsigned __int8 *a3)
{
  bool result; // al
  unsigned __int8 *v4; // rcx
  int v5; // eax
  unsigned __int8 *v6; // r12
  __int64 v7; // rdx
  __int64 v8; // r13
  unsigned int v9; // r14d
  __int64 v10; // rax
  _BYTE *v11; // rax
  int v12; // eax
  unsigned __int8 *v13; // [rsp-30h] [rbp-30h]
  unsigned __int8 *v14; // [rsp-30h] [rbp-30h]
  unsigned __int8 *v15; // [rsp-30h] [rbp-30h]

  if ( a2 + 29 != *a3 )
    return 0;
  v4 = (unsigned __int8 *)*((_QWORD *)a3 - 8);
  v5 = *(_DWORD *)(a1 + 16) + 29;
  if ( *v4 == v5 && *((_QWORD *)v4 - 8) == **(_QWORD **)a1 )
  {
    v8 = *((_QWORD *)v4 - 4);
    if ( !v8 )
      BUG();
    if ( *(_BYTE *)v8 == 17 )
      goto LABEL_15;
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v8 + 8) + 8LL) - 17 <= 1 && *(_BYTE *)v8 <= 0x15u )
    {
      v14 = a3;
      v11 = sub_AD7630(v8, 0, (__int64)a3);
      a3 = v14;
      v8 = (__int64)v11;
      if ( !v11 || *v11 != 17 )
      {
        v6 = (unsigned __int8 *)*((_QWORD *)v14 - 4);
        v5 = *(_DWORD *)(a1 + 16) + 29;
        goto LABEL_5;
      }
LABEL_15:
      v9 = *(_DWORD *)(v8 + 32);
      v6 = (unsigned __int8 *)*((_QWORD *)a3 - 4);
      if ( v9 > 0x40 )
      {
        v15 = a3;
        v12 = sub_C444A0(v8 + 24);
        a3 = v15;
        if ( v9 - v12 > 0x40 )
        {
LABEL_18:
          v5 = *(_DWORD *)(a1 + 16) + 29;
          goto LABEL_5;
        }
        v10 = **(_QWORD **)(v8 + 24);
      }
      else
      {
        v10 = *(_QWORD *)(v8 + 24);
      }
      if ( *(_QWORD *)(a1 + 8) == v10 && v6 )
      {
        **(_QWORD **)(a1 + 24) = v6;
        return 1;
      }
      goto LABEL_18;
    }
  }
  v6 = (unsigned __int8 *)*((_QWORD *)a3 - 4);
LABEL_5:
  if ( *v6 != v5 )
    return 0;
  if ( *((_QWORD *)v6 - 8) != **(_QWORD **)a1 )
    return 0;
  v13 = a3;
  result = sub_F17ED0((_QWORD *)(a1 + 8), *((_QWORD *)v6 - 4));
  if ( !result )
    return 0;
  v7 = *((_QWORD *)v13 - 8);
  if ( !v7 )
    return 0;
  **(_QWORD **)(a1 + 24) = v7;
  return result;
}
