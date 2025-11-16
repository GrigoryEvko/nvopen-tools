// Function: sub_11327B0
// Address: 0x11327b0
//
bool __fastcall sub_11327B0(__int64 a1, int a2, unsigned __int8 *a3)
{
  _BYTE *v5; // r13
  __int64 v6; // r14
  unsigned int v7; // r15d
  bool v8; // al
  _BYTE *v9; // r13
  unsigned int v10; // ebx
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 v13; // rdx
  _BYTE *v14; // rax
  bool v15; // dl
  __int64 v16; // rdx
  _BYTE *v17; // rax
  unsigned int v18; // r15d
  _BYTE *v19; // rax
  int v20; // [rsp-40h] [rbp-40h]
  bool v21; // [rsp-39h] [rbp-39h]

  if ( a2 + 29 != *a3 )
    return 0;
  v5 = (_BYTE *)*((_QWORD *)a3 - 8);
  if ( *v5 != 44 )
    return 0;
  v6 = *((_QWORD *)v5 - 8);
  if ( *(_BYTE *)v6 == 17 )
  {
    v7 = *(_DWORD *)(v6 + 32);
    if ( v7 <= 0x40 )
      v8 = *(_QWORD *)(v6 + 24) == 0;
    else
      v8 = v7 == (unsigned int)sub_C444A0(v6 + 24);
    if ( !v8 )
      return 0;
  }
  else
  {
    v12 = *(_QWORD *)(v6 + 8);
    v13 = (unsigned int)*(unsigned __int8 *)(v12 + 8) - 17;
    if ( (unsigned int)v13 > 1 || *(_BYTE *)v6 > 0x15u )
      return 0;
    v14 = sub_AD7630(*((_QWORD *)v5 - 8), 0, v13);
    if ( !v14 || *v14 != 17 )
    {
      if ( *(_BYTE *)(v12 + 8) == 17 )
      {
        v20 = *(_DWORD *)(v12 + 32);
        if ( v20 )
        {
          v15 = 0;
          v18 = 0;
          while ( 1 )
          {
            v21 = v15;
            v19 = (_BYTE *)sub_AD69F0((unsigned __int8 *)v6, v18);
            if ( !v19 )
              break;
            v15 = v21;
            if ( *v19 != 13 )
            {
              if ( *v19 != 17 )
                break;
              v15 = sub_9867B0((__int64)(v19 + 24));
              if ( !v15 )
                break;
            }
            if ( v20 == ++v18 )
              goto LABEL_22;
          }
        }
      }
      return 0;
    }
    v15 = sub_9867B0((__int64)(v14 + 24));
LABEL_22:
    if ( !v15 )
      return 0;
  }
  if ( *(_QWORD *)a1 )
    **(_QWORD **)a1 = v6;
  if ( *((_QWORD *)v5 - 4) != **(_QWORD **)(a1 + 8) )
    return 0;
  v9 = (_BYTE *)*((_QWORD *)a3 - 4);
  if ( !v9 )
    BUG();
  if ( *v9 != 17 )
  {
    v16 = (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v9 + 1) + 8LL) - 17;
    if ( (unsigned int)v16 > 1 )
      return 0;
    if ( *v9 > 0x15u )
      return 0;
    v17 = sub_AD7630(*((_QWORD *)a3 - 4), 0, v16);
    v9 = v17;
    if ( !v17 || *v17 != 17 )
      return 0;
  }
  v10 = *((_DWORD *)v9 + 8);
  if ( v10 > 0x40 )
  {
    if ( v10 - (unsigned int)sub_C444A0((__int64)(v9 + 24)) <= 0x40 )
    {
      v11 = **((_QWORD **)v9 + 3);
      return *(_QWORD *)(a1 + 16) == v11;
    }
    return 0;
  }
  v11 = *((_QWORD *)v9 + 3);
  return *(_QWORD *)(a1 + 16) == v11;
}
