// Function: sub_13CDFA0
// Address: 0x13cdfa0
//
__int64 __fastcall sub_13CDFA0(_QWORD *a1, __int64 a2, _QWORD *a3)
{
  unsigned __int64 v5; // rax
  unsigned int v6; // r13d
  bool v7; // al
  __int64 v8; // rdi
  __int64 result; // rax
  __int64 v11; // rcx
  int v12; // eax
  __int64 *v13; // rdx
  int v14; // r12d
  bool v15; // zf
  __int64 v16; // rax
  unsigned int v17; // r14d
  bool v18; // al
  unsigned __int64 v19; // rdx
  void *v20; // rcx
  int v21; // r14d
  unsigned int v22; // r15d
  __int64 v23; // rax
  char v24; // cl
  unsigned int v25; // esi
  int v26; // [rsp+Ch] [rbp-34h]

  v5 = *((unsigned __int8 *)a1 + 16);
  if ( (_BYTE)v5 == 13 )
  {
    v6 = *((_DWORD *)a1 + 8);
    if ( v6 <= 0x40 )
      v7 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v6) == a1[3];
    else
      v7 = v6 == (unsigned int)sub_16A58F0(a1 + 3);
    if ( v7 )
    {
LABEL_5:
      v8 = *a1;
      return sub_15A04A0(v8);
    }
    goto LABEL_18;
  }
  if ( *(_BYTE *)(*a1 + 8LL) == 16 && (unsigned __int8)v5 <= 0x10u )
  {
    v16 = sub_15A1020(a1);
    if ( v16 && *(_BYTE *)(v16 + 16) == 13 )
    {
      v17 = *(_DWORD *)(v16 + 32);
      if ( v17 <= 0x40 )
        v18 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v17) == *(_QWORD *)(v16 + 24);
      else
        v18 = v17 == (unsigned int)sub_16A58F0(v16 + 24);
      if ( v18 )
        goto LABEL_5;
    }
    else
    {
      v8 = *a1;
      v21 = *(_QWORD *)(*a1 + 32LL);
      if ( !v21 )
        return sub_15A04A0(v8);
      v22 = 0;
      while ( 1 )
      {
        v23 = sub_15A0A60(a1, v22);
        if ( !v23 )
          break;
        v24 = *(_BYTE *)(v23 + 16);
        if ( v24 != 9 )
        {
          if ( v24 != 13 )
            break;
          v25 = *(_DWORD *)(v23 + 32);
          if ( v25 <= 0x40 )
          {
            if ( *(_QWORD *)(v23 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v25) )
              break;
          }
          else
          {
            v26 = *(_DWORD *)(v23 + 32);
            if ( v26 != (unsigned int)sub_16A58F0(v23 + 24) )
              break;
          }
        }
        if ( v21 == ++v22 )
          goto LABEL_5;
      }
    }
    v5 = *((unsigned __int8 *)a1 + 16);
    if ( (unsigned __int8)v5 > 0x17u )
      goto LABEL_10;
LABEL_28:
    if ( (_BYTE)v5 != 5 )
      goto LABEL_18;
    v19 = *((unsigned __int16 *)a1 + 9);
    if ( (unsigned __int16)v19 > 0x17u )
      goto LABEL_18;
    v20 = &loc_80A800;
    v12 = (unsigned __int16)v19;
    if ( !_bittest64((const __int64 *)&v20, v19) )
      goto LABEL_18;
    goto LABEL_13;
  }
  if ( (unsigned __int8)v5 <= 0x17u )
    goto LABEL_28;
LABEL_10:
  if ( (unsigned __int8)v5 > 0x2Fu )
    goto LABEL_18;
  v11 = 0x80A800000000LL;
  if ( !_bittest64(&v11, v5) )
    goto LABEL_18;
  v12 = (unsigned __int8)v5 - 24;
LABEL_13:
  if ( v12 == 23 && (*((_BYTE *)a1 + 17) & 4) != 0 )
  {
    if ( (*((_BYTE *)a1 + 23) & 0x40) != 0 )
    {
      v13 = (__int64 *)*(a1 - 1);
      result = *v13;
      if ( *v13 )
      {
LABEL_17:
        if ( a2 == v13[3] )
          return result;
      }
    }
    else
    {
      v13 = &a1[-3 * (*((_DWORD *)a1 + 5) & 0xFFFFFFF)];
      result = *v13;
      if ( *v13 )
        goto LABEL_17;
    }
  }
LABEL_18:
  v14 = sub_14C23D0(a1, *a3, 0, a3[3], a3[4], a3[2]);
  v15 = v14 == (unsigned int)sub_16431D0(*a1);
  result = 0;
  if ( v15 )
    return (__int64)a1;
  return result;
}
