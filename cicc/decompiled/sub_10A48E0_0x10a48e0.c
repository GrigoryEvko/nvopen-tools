// Function: sub_10A48E0
// Address: 0x10a48e0
//
__int64 __fastcall sub_10A48E0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  __int64 *v4; // rdx
  _BYTE *v5; // r14
  unsigned __int64 v6; // rax
  unsigned int v7; // edx
  __int64 v8; // rax
  _BYTE *v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rbx
  __int64 v13; // rbx
  unsigned int v14; // r14d
  bool v15; // al
  __int64 *v16; // rax
  __int64 v17; // r14
  __int64 v18; // rdx
  _BYTE *v19; // rax
  unsigned int v20; // r14d
  char v21; // r15
  unsigned int v22; // r14d
  __int64 v23; // rax
  unsigned int v24; // r15d
  int v25; // [rsp+Ch] [rbp-44h]

  if ( *(_BYTE *)a2 != 86 )
    return 0;
  v4 = (*(_BYTE *)(a2 + 7) & 0x40) != 0
     ? *(__int64 **)(a2 - 8)
     : (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v5 = (_BYTE *)*v4;
  if ( *(_BYTE *)*v4 != 82 )
    return 0;
  v6 = sub_B53900(*v4);
  sub_B53630(v6, *(_QWORD *)a1);
  v2 = v7;
  if ( !(_BYTE)v7 )
    return 0;
  v8 = *((_QWORD *)v5 - 8);
  if ( !v8 )
    return 0;
  **(_QWORD **)(a1 + 8) = v8;
  v9 = (_BYTE *)*((_QWORD *)v5 - 4);
  if ( *v9 != 17 )
    return 0;
  **(_QWORD **)(a1 + 16) = v9;
  v10 = (*(_BYTE *)(a2 + 7) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v11 = *(_QWORD *)(v10 + 32);
  if ( !v11 )
    return 0;
  **(_QWORD **)(a1 + 24) = v11;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v12 = *(_QWORD *)(a2 - 8);
  else
    v12 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v13 = *(_QWORD *)(v12 + 64);
  if ( *(_BYTE *)v13 == 17 )
  {
    v14 = *(_DWORD *)(v13 + 32);
    if ( v14 <= 0x40 )
      v15 = *(_QWORD *)(v13 + 24) == 0;
    else
      v15 = v14 == (unsigned int)sub_C444A0(v13 + 24);
LABEL_18:
    if ( v15 )
      goto LABEL_19;
    return 0;
  }
  v17 = *(_QWORD *)(v13 + 8);
  v18 = (unsigned int)*(unsigned __int8 *)(v17 + 8) - 17;
  if ( (unsigned int)v18 > 1 || *(_BYTE *)v13 > 0x15u )
    return 0;
  v19 = sub_AD7630(v13, 0, v18);
  if ( !v19 || *v19 != 17 )
  {
    if ( *(_BYTE *)(v17 + 8) == 17 )
    {
      v25 = *(_DWORD *)(v17 + 32);
      if ( v25 )
      {
        v21 = 0;
        v22 = 0;
        while ( 1 )
        {
          v23 = sub_AD69F0((unsigned __int8 *)v13, v22);
          if ( !v23 )
            break;
          if ( *(_BYTE *)v23 != 13 )
          {
            if ( *(_BYTE *)v23 != 17 )
              return 0;
            v24 = *(_DWORD *)(v23 + 32);
            if ( v24 <= 0x40 )
            {
              if ( *(_QWORD *)(v23 + 24) )
                return 0;
            }
            else if ( v24 != (unsigned int)sub_C444A0(v23 + 24) )
            {
              return 0;
            }
            v21 = v2;
          }
          if ( v25 == ++v22 )
          {
            if ( v21 )
              goto LABEL_19;
            return 0;
          }
        }
      }
    }
    return 0;
  }
  v20 = *((_DWORD *)v19 + 8);
  if ( v20 > 0x40 )
  {
    v15 = v20 == (unsigned int)sub_C444A0((__int64)(v19 + 24));
    goto LABEL_18;
  }
  if ( *((_QWORD *)v19 + 3) )
    return 0;
LABEL_19:
  v16 = *(__int64 **)(a1 + 32);
  if ( v16 )
    *v16 = v13;
  return v2;
}
