// Function: sub_2755220
// Address: 0x2755220
//
bool __fastcall sub_2755220(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r12
  unsigned __int64 v4; // rbx
  __int64 *v5; // rsi
  __int64 v6; // r13
  _BYTE *v8; // r14
  unsigned __int64 v9; // rax
  char v10; // dl
  char v11; // r15
  __int64 v12; // r13
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned int v15; // r14d
  bool v16; // al
  __int64 v17; // r14
  _BYTE *v18; // rax
  char v19; // dl
  unsigned int v20; // r13d
  int v21; // eax
  unsigned int v22; // r14d
  __int64 v23; // rax
  char v24; // [rsp+8h] [rbp-78h]
  int v25; // [rsp+8h] [rbp-78h]
  int v26; // [rsp+Ch] [rbp-74h]

  v2 = *(_QWORD *)(a1 + 40);
  v3 = *(_QWORD *)(a2 + 40);
  if ( v2 != v3 )
  {
    v4 = *(_QWORD *)(v2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    v5 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    if ( v4 == v2 + 48 )
      goto LABEL_46;
    if ( !v4 )
      BUG();
    if ( (unsigned int)*(unsigned __int8 *)(v4 - 24) - 30 > 0xA )
LABEL_46:
      BUG();
    v6 = *v5;
    if ( *(_BYTE *)(v4 - 24) != 31 )
      return 0;
    if ( (*(_DWORD *)(v4 - 20) & 0x7FFFFFF) != 3 )
      return 0;
    v8 = *(_BYTE **)(v4 - 120);
    if ( *v8 != 82 )
      return 0;
    v9 = sub_B53900(*(_QWORD *)(v4 - 120));
    sub_B53630(v9, 32);
    v11 = v10;
    if ( !v10 )
      return 0;
    if ( v6 != *((_QWORD *)v8 - 8) )
      return 0;
    v12 = *((_QWORD *)v8 - 4);
    if ( *(_BYTE *)v12 > 0x15u )
      return 0;
    if ( !sub_AC30F0(*((_QWORD *)v8 - 4)) )
    {
      if ( *(_BYTE *)v12 == 17 )
      {
        v15 = *(_DWORD *)(v12 + 32);
        if ( v15 <= 0x40 )
          v16 = *(_QWORD *)(v12 + 24) == 0;
        else
          v16 = v15 == (unsigned int)sub_C444A0(v12 + 24);
        if ( !v16 )
          return 0;
      }
      else
      {
        v17 = *(_QWORD *)(v12 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v17 + 8) - 17 > 1 )
          return 0;
        v18 = sub_AD7630(v12, 0, v13);
        v19 = 0;
        if ( v18 && *v18 == 17 )
        {
          v20 = *((_DWORD *)v18 + 8);
          if ( v20 <= 0x40 )
          {
            if ( *((_QWORD *)v18 + 3) )
              return 0;
          }
          else if ( v20 != (unsigned int)sub_C444A0((__int64)(v18 + 24)) )
          {
            return 0;
          }
        }
        else
        {
          if ( *(_BYTE *)(v17 + 8) != 17 )
            return 0;
          v21 = *(_DWORD *)(v17 + 32);
          v22 = 0;
          v26 = v21;
          while ( v26 != v22 )
          {
            v24 = v19;
            v23 = sub_AD69F0((unsigned __int8 *)v12, v22);
            if ( !v23 )
              return 0;
            v19 = v24;
            if ( *(_BYTE *)v23 != 13 )
            {
              if ( *(_BYTE *)v23 != 17 )
                return 0;
              if ( *(_DWORD *)(v23 + 32) <= 0x40u )
              {
                if ( *(_QWORD *)(v23 + 24) )
                  return 0;
              }
              else
              {
                v25 = *(_DWORD *)(v23 + 32);
                if ( v25 != (unsigned int)sub_C444A0(v23 + 24) )
                  return 0;
              }
              v19 = v11;
            }
            ++v22;
          }
          if ( !v19 )
            return 0;
        }
      }
    }
    if ( *(_QWORD *)(v4 - 56) )
    {
      v14 = *(_QWORD *)(v4 - 88);
      if ( v14 )
        return v3 == v14;
    }
    return 0;
  }
  return 1;
}
