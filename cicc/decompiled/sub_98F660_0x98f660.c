// Function: sub_98F660
// Address: 0x98f660
//
bool __fastcall sub_98F660(unsigned __int8 *a1, unsigned __int8 *a2, char a3, char a4)
{
  __int64 v8; // rdx
  unsigned int v9; // r14d
  bool v10; // al
  __int64 v11; // rcx
  unsigned int v12; // r14d
  bool v13; // al
  unsigned __int8 *v14; // rax
  __int64 v15; // r14
  __int64 v16; // rax
  __int64 v17; // rdx
  unsigned int v18; // r14d
  unsigned __int64 v19; // rax
  __int64 v20; // rcx
  int v21; // eax
  __int64 v22; // rsi
  __int64 v23; // rcx
  unsigned __int64 v24; // rax
  __int64 v25; // rdi
  int v26; // eax
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r14
  _BYTE *v30; // rax
  __int64 v31; // rcx
  bool v32; // dl
  unsigned int v33; // r14d
  _BYTE *v34; // rax
  bool v35; // al
  unsigned int v36; // r14d
  _BYTE *v37; // rax
  bool v38; // al
  bool v39; // [rsp+0h] [rbp-40h]
  int v40; // [rsp+0h] [rbp-40h]
  int v41; // [rsp+4h] [rbp-3Ch]
  bool v42; // [rsp+4h] [rbp-3Ch]
  __int64 v43; // [rsp+8h] [rbp-38h]
  __int64 v44; // [rsp+8h] [rbp-38h]
  __int64 v45; // [rsp+8h] [rbp-38h]
  __int64 v46; // [rsp+8h] [rbp-38h]

  if ( *a1 != 44 )
    goto LABEL_2;
  v8 = *((_QWORD *)a1 - 8);
  if ( *(_BYTE *)v8 == 17 )
  {
    v9 = *(_DWORD *)(v8 + 32);
    if ( v9 <= 0x40 )
      v10 = *(_QWORD *)(v8 + 24) == 0;
    else
      v10 = v9 == (unsigned int)sub_C444A0(v8 + 24);
  }
  else
  {
    v15 = *(_QWORD *)(v8 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v15 + 8) - 17 > 1 || *(_BYTE *)v8 > 0x15u )
      goto LABEL_2;
    v43 = *((_QWORD *)a1 - 8);
    v16 = sub_AD7630(v43, 0);
    v17 = v43;
    if ( !v16 || *(_BYTE *)v16 != 17 )
    {
      if ( *(_BYTE *)(v15 + 8) == 17 )
      {
        v41 = *(_DWORD *)(v15 + 32);
        if ( v41 )
        {
          v39 = 0;
          v33 = 0;
          while ( 1 )
          {
            v45 = v17;
            v34 = (_BYTE *)sub_AD69F0(v17, v33);
            if ( !v34 )
              break;
            v17 = v45;
            if ( *v34 != 13 )
            {
              if ( *v34 != 17 )
                break;
              v35 = sub_9867B0((__int64)(v34 + 24));
              v17 = v45;
              v39 = v35;
              if ( !v35 )
                break;
            }
            if ( v41 == ++v33 )
            {
              if ( v39 )
                goto LABEL_10;
              goto LABEL_2;
            }
          }
        }
      }
      goto LABEL_2;
    }
    v18 = *(_DWORD *)(v16 + 32);
    if ( v18 <= 0x40 )
      v10 = *(_QWORD *)(v16 + 24) == 0;
    else
      v10 = v18 == (unsigned int)sub_C444A0(v16 + 24);
  }
  if ( v10 )
  {
LABEL_10:
    if ( a2 == *((unsigned __int8 **)a1 - 4)
      && (!a3 || (unsigned __int8)sub_B44900(a1))
      && (a4 || (unsigned __int8)sub_AC30F0(*((_QWORD *)a1 - 8))) )
    {
      return 1;
    }
  }
LABEL_2:
  if ( *a2 != 44 )
    goto LABEL_3;
  v11 = *((_QWORD *)a2 - 8);
  if ( *(_BYTE *)v11 == 17 )
  {
    v12 = *(_DWORD *)(v11 + 32);
    if ( v12 <= 0x40 )
      v13 = *(_QWORD *)(v11 + 24) == 0;
    else
      v13 = v12 == (unsigned int)sub_C444A0(v11 + 24);
    if ( !v13 )
      goto LABEL_3;
  }
  else
  {
    v29 = *(_QWORD *)(v11 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v29 + 8) - 17 > 1 || *(_BYTE *)v11 > 0x15u )
      goto LABEL_3;
    v44 = *((_QWORD *)a2 - 8);
    v30 = (_BYTE *)sub_AD7630(v44, 0);
    v31 = v44;
    if ( !v30 || *v30 != 17 )
    {
      if ( *(_BYTE *)(v29 + 8) == 17 )
      {
        v40 = *(_DWORD *)(v29 + 32);
        if ( v40 )
        {
          v32 = 0;
          v36 = 0;
          while ( 1 )
          {
            v42 = v32;
            v46 = v31;
            v37 = (_BYTE *)sub_AD69F0(v31, v36);
            if ( !v37 )
              break;
            v31 = v46;
            v32 = v42;
            if ( *v37 != 13 )
            {
              if ( *v37 != 17 )
                break;
              v38 = sub_9867B0((__int64)(v37 + 24));
              v31 = v46;
              v32 = v38;
              if ( !v38 )
                break;
            }
            if ( v40 == ++v36 )
              goto LABEL_60;
          }
        }
      }
      goto LABEL_3;
    }
    v32 = sub_9867B0((__int64)(v30 + 24));
LABEL_60:
    if ( !v32 )
      goto LABEL_3;
  }
  v14 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
  if ( a1 == v14 && v14 )
  {
    if ( a3 && !(unsigned __int8)sub_B44900(a2) )
      goto LABEL_32;
    if ( a4 || (unsigned __int8)sub_AC30F0(*((_QWORD *)a2 - 8)) )
      return 1;
  }
LABEL_3:
  if ( !a3 )
  {
    if ( *a1 == 44 )
    {
      v27 = *((_QWORD *)a1 - 8);
      if ( v27 )
      {
        v28 = *((_QWORD *)a1 - 4);
        if ( v28 )
        {
          if ( *a2 == 44 && v28 == *((_QWORD *)a2 - 8) )
            return *((_QWORD *)a2 - 4) == v27;
        }
      }
    }
    return 0;
  }
LABEL_32:
  v19 = *a1;
  if ( (unsigned __int8)v19 <= 0x1Cu )
  {
    if ( (_BYTE)v19 != 5 )
      return 0;
    v21 = *((unsigned __int16 *)a1 + 1);
    if ( (*((_WORD *)a1 + 1) & 0xFFF7) != 0x11 && (v21 & 0xFFFD) != 0xD )
      return 0;
  }
  else
  {
    if ( (unsigned __int8)v19 > 0x36u )
      return 0;
    v20 = 0x40540000000000LL;
    if ( !_bittest64(&v20, v19) )
      return 0;
    v21 = (unsigned __int8)v19 - 29;
  }
  if ( v21 != 15 )
    return 0;
  if ( (a1[1] & 4) == 0 )
    return 0;
  v22 = *((_QWORD *)a1 - 8);
  if ( !v22 )
    return 0;
  v23 = *((_QWORD *)a1 - 4);
  if ( !v23 )
    return 0;
  v24 = *a2;
  if ( (unsigned __int8)v24 > 0x1Cu )
  {
    if ( (unsigned __int8)v24 > 0x36u )
      return 0;
    v25 = 0x40540000000000LL;
    if ( !_bittest64(&v25, v24) )
      return 0;
    v26 = (unsigned __int8)v24 - 29;
LABEL_44:
    if ( v26 == 15 && (a2[1] & 4) != 0 && v23 == *((_QWORD *)a2 - 8) )
      return *((_QWORD *)a2 - 4) == v22;
    return 0;
  }
  if ( (_BYTE)v24 != 5 )
    return 0;
  v26 = *((unsigned __int16 *)a2 + 1);
  if ( (*((_WORD *)a2 + 1) & 0xFFFD) == 0xD || (v26 & 0xFFF7) == 0x11 )
    goto LABEL_44;
  return 0;
}
