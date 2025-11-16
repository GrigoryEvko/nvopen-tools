// Function: sub_14AB9D0
// Address: 0x14ab9d0
//
bool __fastcall sub_14AB9D0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  unsigned __int8 v4; // bl
  unsigned __int8 v5; // al
  char v6; // al
  __int64 *v8; // r13
  int v9; // r14d
  int *v10; // rax
  int v11; // eax
  int v12; // eax
  int v13; // eax
  int v14; // r13d
  char v15; // r15
  unsigned int v16; // r14d
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rbx
  __int64 v21; // rbx
  char v22; // al
  __int64 *v23; // rax
  _QWORD *v24; // rax
  unsigned int v25; // r14d
  unsigned int v26; // r15d
  __int64 v27; // rax
  unsigned int v28; // r14d
  __int64 v29; // rax
  __int64 v30; // rax
  unsigned int v31; // eax
  __int64 v32; // rcx
  _QWORD *v33; // rax
  __int64 **v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rcx
  _QWORD *v37; // rax
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // rax
  __int64 *v42; // rax
  __int64 v43; // rdx
  unsigned int v44; // esi

  while ( 2 )
  {
    v4 = a3;
    v5 = *(_BYTE *)(a1 + 16);
    if ( v5 == 14 )
    {
      if ( *(_QWORD *)(a1 + 32) == sub_16982C0(a1, a2, a3, a4) )
      {
        v6 = *(_BYTE *)(*(_QWORD *)(a1 + 40) + 26LL);
        if ( (v6 & 8) == 0 )
          return 1;
      }
      else
      {
        v6 = *(_BYTE *)(a1 + 50);
        if ( (v6 & 8) == 0 )
          return 1;
      }
      if ( v4 )
        return 0;
      return (v6 & 7) == 3;
    }
    v8 = a2;
    v9 = a4;
    if ( v5 <= 0x10u && *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 16 )
    {
      v14 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
      if ( !v14 )
        return 1;
      v15 = a3;
      v16 = 0;
      while ( 1 )
      {
        v17 = sub_15A0A60(a1, v16);
        v20 = v17;
        if ( !v17 || *(_BYTE *)(v17 + 16) != 14 )
          break;
        v21 = *(_QWORD *)(v17 + 32) == sub_16982C0(a1, v16, v18, v19) ? *(_QWORD *)(v20 + 40) + 8LL : v20 + 32;
        v22 = *(_BYTE *)(v21 + 18);
        if ( (v22 & 8) != 0 && (v15 || (v22 & 7) != 3) )
          break;
        if ( ++v16 == v14 )
          return 1;
      }
      return 0;
    }
    v10 = (int *)sub_16D40F0(qword_4FBB370);
    if ( v10 )
      v11 = *v10;
    else
      v11 = qword_4FBB370[2];
    if ( v9 == v11 )
      return 0;
    v12 = *(unsigned __int8 *)(a1 + 16);
    if ( (unsigned __int8)v12 <= 0x17u )
    {
      if ( (_BYTE)v12 != 5 )
        return 0;
      v13 = *(unsigned __int16 *)(a1 + 18);
    }
    else
    {
      v13 = v12 - 24;
    }
    switch ( v13 )
    {
      case 12:
      case 19:
      case 22:
        v24 = (_QWORD *)sub_13CF970(a1);
        goto LABEL_34;
      case 16:
        v24 = (_QWORD *)sub_13CF970(a1);
        if ( *v24 == v24[3] && (!v4 || (*(_BYTE *)(a1 + 17) & 4) != 0) )
          return 1;
LABEL_34:
        v25 = v9 + 1;
        v26 = v4;
        goto LABEL_35;
      case 41:
        return 1;
      case 43:
      case 44:
      case 59:
        v23 = (__int64 *)sub_13CF970(a1);
        a4 = (unsigned int)(v9 + 1);
        a3 = v4;
        a1 = *v23;
        continue;
      case 54:
        v31 = sub_14AB140(a1 | 4, a2);
        if ( v31 == 132 )
        {
          v34 = (__int64 **)sub_13CF970(a1);
          if ( sub_14AB850(*v34, (__int64)a2, v35, v36) )
          {
            v37 = (_QWORD *)sub_13CF970(a1);
            if ( (unsigned __int8)sub_14AB9D0(*v37, a2, v4, (unsigned int)(v9 + 1)) )
              return 1;
          }
          v38 = sub_13CF970(a1);
          if ( !sub_14AB850(*(__int64 **)(v38 + 24), (__int64)a2, v39, v40) )
            return 0;
          v41 = sub_13CF970(a1);
          a4 = (unsigned int)(v9 + 1);
          a3 = v4;
          a1 = *(_QWORD *)(v41 + 24);
          continue;
        }
        if ( v31 > 0x84 )
        {
          switch ( v31 )
          {
            case 0x93u:
              v42 = (__int64 *)sub_13CF970(a1);
              v43 = v42[3];
              if ( *(_BYTE *)(v43 + 16) == 13 )
              {
                v44 = *(_DWORD *)(v43 + 32);
                if ( v44 <= 0x40
                  && (((__int64)(*(_QWORD *)(v43 + 24) << (64 - (unsigned __int8)v44)) >> (64 - (unsigned __int8)v44))
                    & 1) == 0 )
                {
                  return 1;
                }
              }
              a1 = *v42;
              a4 = (unsigned int)(v9 + 1);
              a3 = v4;
              a2 = v8;
              break;
            case 0xC4u:
              if ( !v4 )
                return 1;
              if ( !(unsigned __int8)sub_15F24B0(a1) )
                return 0;
              if ( (unsigned __int8)sub_15F24C0(a1) )
                return 1;
              return sub_14AB3F0(*(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)), a2, 0, v32);
            case 0x8Bu:
              v25 = v9 + 1;
              v26 = v4;
              v24 = (_QWORD *)sub_13CF970(a1);
LABEL_35:
              if ( !(unsigned __int8)sub_14AB9D0(*v24, a2, v26, v25) )
                return 0;
              v27 = sub_13CF970(a1);
              a4 = v25;
              a3 = v26;
              a1 = *(_QWORD *)(v27 + 24);
              break;
            default:
              return 0;
          }
          continue;
        }
        if ( v31 == 96 )
          return 1;
        if ( v31 > 0x60 )
        {
          if ( v31 - 99 > 1 )
            return 0;
          v33 = (_QWORD *)sub_13CF970(a1);
          if ( v33[3] != *v33 || v4 && (*(_BYTE *)(a1 + 17) & 4) == 0 )
            return 0;
          a1 = v33[6];
          a4 = (unsigned int)(v9 + 1);
          a3 = v4;
          continue;
        }
        return v31 - 54 <= 1;
      case 55:
        v28 = v9 + 1;
        v29 = sub_13CF970(a1);
        if ( !(unsigned __int8)sub_14AB9D0(*(_QWORD *)(v29 + 24), a2, v4, v28) )
          return 0;
        v30 = sub_13CF970(a1);
        a4 = v28;
        a3 = v4;
        a1 = *(_QWORD *)(v30 + 48);
        continue;
      default:
        return 0;
    }
  }
}
