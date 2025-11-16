// Function: sub_1A6E4A0
// Address: 0x1a6e4a0
//
__int64 __fastcall sub_1A6E4A0(
        __int64 a1,
        unsigned __int64 a2,
        double a3,
        double a4,
        double a5,
        __int64 a6,
        __int64 a7)
{
  unsigned __int8 v8; // al
  __int64 v10; // rbx
  __int64 v11; // r13
  _QWORD *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  _BYTE *v15; // r12
  __int64 *v16; // rbx
  unsigned __int8 v17; // al
  unsigned int v18; // r13d
  bool v19; // al
  __int64 v20; // rax
  unsigned int v21; // ebx
  bool v22; // al
  unsigned __int8 v23; // al
  unsigned int v24; // r13d
  bool v25; // al
  unsigned __int64 v26; // rax
  __int64 v27; // rdi
  unsigned __int64 v28; // r12
  __int64 v29; // rdx
  __int64 v30; // rax
  unsigned int v31; // r12d
  int v32; // r13d
  unsigned int v33; // r14d
  __int64 v34; // rax
  unsigned int v35; // esi
  bool v36; // al
  unsigned int v37; // r13d
  __int64 v38; // rax
  char v39; // cl
  unsigned int v40; // r14d
  unsigned int v42; // [rsp+Ch] [rbp-64h]
  int v43; // [rsp+Ch] [rbp-64h]
  _QWORD v44[2]; // [rsp+10h] [rbp-60h] BYREF
  __int64 v45[2]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v46; // [rsp+30h] [rbp-40h]

  v8 = *(_BYTE *)(a1 + 16);
  if ( v8 <= 0x10u )
    return sub_15A2B00((__int64 *)a1, a3, a4, a5);
  if ( v8 != 52 )
    goto LABEL_4;
  v15 = *(_BYTE **)(a1 - 48);
  v16 = *(__int64 **)(a1 - 24);
  if ( v15 )
  {
    v17 = *((_BYTE *)v16 + 16);
    if ( v17 == 13 )
    {
      v18 = *((_DWORD *)v16 + 8);
      if ( v18 <= 0x40 )
      {
        a7 = 64 - v18;
        v19 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v18) == v16[3];
      }
      else
      {
        v19 = v18 == (unsigned int)sub_16A58F0((__int64)(v16 + 3));
      }
      if ( v19 )
        return (__int64)v15;
      goto LABEL_28;
    }
    if ( *(_BYTE *)(*v16 + 8) != 16 || v17 > 0x10u )
      goto LABEL_28;
    v20 = sub_15A1020(*(_BYTE **)(a1 - 24), a2, *v16, a7);
    if ( v20 && *(_BYTE *)(v20 + 16) == 13 )
    {
      v21 = *(_DWORD *)(v20 + 32);
      if ( v21 <= 0x40 )
      {
        a7 = 64 - v21;
        v22 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v21) == *(_QWORD *)(v20 + 24);
      }
      else
      {
        v22 = v21 == (unsigned int)sub_16A58F0(v20 + 24);
      }
      if ( v22 )
        return (__int64)v15;
    }
    else
    {
      v32 = *(_QWORD *)(*v16 + 32);
      if ( !v32 )
        return (__int64)v15;
      v33 = 0;
      while ( 1 )
      {
        a2 = v33;
        v34 = sub_15A0A60((__int64)v16, v33);
        if ( !v34 )
          break;
        a7 = *(unsigned __int8 *)(v34 + 16);
        if ( (_BYTE)a7 != 9 )
        {
          if ( (_BYTE)a7 != 13 )
            break;
          v35 = *(_DWORD *)(v34 + 32);
          if ( v35 <= 0x40 )
          {
            a7 = 64 - v35;
            a2 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v35);
            v36 = a2 == *(_QWORD *)(v34 + 24);
          }
          else
          {
            v42 = *(_DWORD *)(v34 + 32);
            a2 = v42;
            v36 = v42 == (unsigned int)sub_16A58F0(v34 + 24);
          }
          if ( !v36 )
            break;
        }
        if ( v32 == ++v33 )
          return (__int64)v15;
      }
    }
    v16 = *(__int64 **)(a1 - 24);
  }
  if ( !v16 )
    goto LABEL_33;
  v15 = *(_BYTE **)(a1 - 48);
LABEL_28:
  v23 = v15[16];
  if ( v23 == 13 )
  {
    v24 = *((_DWORD *)v15 + 8);
    if ( v24 <= 0x40 )
      v25 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v24) == *((_QWORD *)v15 + 3);
    else
      v25 = v24 == (unsigned int)sub_16A58F0((__int64)(v15 + 24));
    goto LABEL_31;
  }
  if ( *(_BYTE *)(*(_QWORD *)v15 + 8LL) != 16 || v23 > 0x10u )
    goto LABEL_33;
  v30 = sub_15A1020(v15, a2, *(_QWORD *)v15, a7);
  if ( !v30 || *(_BYTE *)(v30 + 16) != 13 )
  {
    v43 = *(_QWORD *)(*(_QWORD *)v15 + 32LL);
    if ( !v43 )
      return (__int64)v16;
    v37 = 0;
    while ( 1 )
    {
      v38 = sub_15A0A60((__int64)v15, v37);
      if ( !v38 )
        break;
      v39 = *(_BYTE *)(v38 + 16);
      if ( v39 != 9 )
      {
        if ( v39 != 13 )
          break;
        v40 = *(_DWORD *)(v38 + 32);
        if ( !(v40 <= 0x40
             ? 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v40) == *(_QWORD *)(v38 + 24)
             : v40 == (unsigned int)sub_16A58F0(v38 + 24)) )
          break;
      }
      if ( v43 == ++v37 )
        return (__int64)v16;
    }
LABEL_33:
    v8 = *(_BYTE *)(a1 + 16);
LABEL_4:
    if ( v8 <= 0x17u )
    {
      v27 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 80LL);
      if ( v27 )
        v27 -= 24;
      v28 = sub_157EBA0(v27);
      v44[0] = sub_1649960(a1);
      v46 = 773;
      v44[1] = v29;
      v45[0] = (__int64)v44;
      v45[1] = (__int64)".inv";
      return sub_15FB630((__int64 *)a1, (__int64)v45, v28);
    }
    else
    {
      v10 = *(_QWORD *)(a1 + 8);
      v11 = *(_QWORD *)(a1 + 40);
      if ( v10 )
      {
        while ( 1 )
        {
          v12 = sub_1648700(v10);
          v15 = v12;
          if ( *((_BYTE *)v12 + 16) > 0x17u && v11 == v12[5] )
          {
            v45[0] = a1;
            if ( sub_1A6DEE0(v45, (unsigned __int64)v12, v13, v14) )
              break;
          }
          v10 = *(_QWORD *)(v10 + 8);
          if ( !v10 )
            goto LABEL_34;
        }
      }
      else
      {
LABEL_34:
        v26 = sub_157EBA0(v11);
        v46 = 257;
        return sub_15FB630((__int64 *)a1, (__int64)v45, v26);
      }
    }
    return (__int64)v15;
  }
  v31 = *(_DWORD *)(v30 + 32);
  if ( v31 <= 0x40 )
    v25 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v31) == *(_QWORD *)(v30 + 24);
  else
    v25 = v31 == (unsigned int)sub_16A58F0(v30 + 24);
LABEL_31:
  if ( !v25 )
    goto LABEL_33;
  return (__int64)v16;
}
