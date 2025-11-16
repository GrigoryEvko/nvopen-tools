// Function: sub_109DB70
// Address: 0x109db70
//
__int64 __fastcall sub_109DB70(_BYTE *a1, _QWORD *a2, __int64 a3, unsigned int a4)
{
  char v7; // al
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // r12
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rsi
  __int64 v18; // rdx
  _BYTE *v19; // rax
  __int64 v20; // rdx
  _BYTE *v21; // rax
  __int64 v22; // rdx
  _BYTE *v23; // rax
  unsigned int v24; // [rsp+Ch] [rbp-34h]
  unsigned int v25; // [rsp+Ch] [rbp-34h]
  unsigned int v26; // [rsp+Ch] [rbp-34h]
  unsigned int v27; // [rsp+Ch] [rbp-34h]
  __int64 v28; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v29; // [rsp+18h] [rbp-28h]

  v7 = *a1;
  if ( (_BYTE)a4 )
  {
    if ( v7 == 49 )
    {
      v9 = *((_QWORD *)a1 - 8);
      if ( v9 )
      {
        *a2 = v9;
        v10 = *((_QWORD *)a1 - 4);
        v11 = v10 + 24;
        if ( *(_BYTE *)v10 == 17 )
        {
LABEL_9:
          if ( *(_DWORD *)(a3 + 8) <= 0x40u && *(_DWORD *)(v11 + 8) <= 0x40u )
          {
            *(_QWORD *)a3 = *(_QWORD *)v11;
            *(_DWORD *)(a3 + 8) = *(_DWORD *)(v11 + 8);
            return a4;
          }
          else
          {
            v24 = a4;
            sub_C43990(a3, v11);
            return v24;
          }
        }
        v25 = a4;
        v18 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v10 + 8) + 8LL) - 17;
        if ( (unsigned int)v18 <= 1 && *(_BYTE *)v10 <= 0x15u )
        {
          v19 = sub_AD7630(v10, 0, v18);
          if ( v19 )
          {
            if ( *v19 == 17 )
            {
              a4 = v25;
              v11 = (__int64)(v19 + 24);
              goto LABEL_9;
            }
          }
        }
      }
    }
    return 0;
  }
  if ( v7 == 48 )
  {
    v15 = *((_QWORD *)a1 - 8);
    if ( !v15 )
      return a4;
    *a2 = v15;
    v16 = *((_QWORD *)a1 - 4);
    v17 = v16 + 24;
    if ( *(_BYTE *)v16 == 17
      || (v20 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v16 + 8) + 8LL) - 17, (unsigned int)v20 <= 1)
      && *(_BYTE *)v16 <= 0x15u
      && (v26 = a4, v21 = sub_AD7630(v16, 0, v20), a4 = v26, v21)
      && (v17 = (__int64)(v21 + 24), *v21 == 17) )
    {
      if ( *(_DWORD *)(a3 + 8) <= 0x40u && *(_DWORD *)(v17 + 8) <= 0x40u )
      {
        *(_QWORD *)a3 = *(_QWORD *)v17;
        *(_DWORD *)(a3 + 8) = *(_DWORD *)(v17 + 8);
        return 1;
      }
      else
      {
        sub_C43990(a3, v17);
        return 1;
      }
    }
    v7 = *a1;
  }
  if ( v7 != 55 )
    return a4;
  v12 = *((_QWORD *)a1 - 8);
  if ( !v12 )
    return a4;
  *a2 = v12;
  v13 = *((_QWORD *)a1 - 4);
  v14 = v13 + 24;
  if ( *(_BYTE *)v13 != 17 )
  {
    v22 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v13 + 8) + 8LL) - 17;
    if ( (unsigned int)v22 <= 1 && *(_BYTE *)v13 <= 0x15u )
    {
      v27 = a4;
      v23 = sub_AD7630(v13, 0, v22);
      a4 = v27;
      if ( v23 )
      {
        if ( *v23 == 17 )
        {
          v14 = (__int64)(v23 + 24);
          goto LABEL_14;
        }
      }
    }
    return a4;
  }
LABEL_14:
  v29 = *(_DWORD *)(v14 + 8);
  if ( v29 > 0x40 )
    sub_C43690((__int64)&v28, 1, 0);
  else
    v28 = 1;
  if ( *(_DWORD *)(a3 + 8) > 0x40u && *(_QWORD *)a3 )
    j_j___libc_free_0_0(*(_QWORD *)a3);
  *(_QWORD *)a3 = v28;
  *(_DWORD *)(a3 + 8) = v29;
  sub_C47AC0(a3, v14);
  return 1;
}
