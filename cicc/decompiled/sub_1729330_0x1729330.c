// Function: sub_1729330
// Address: 0x1729330
//
unsigned __int8 *__fastcall sub_1729330(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _BYTE *v5; // r10
  _QWORD *v6; // r15
  _QWORD **v7; // r12
  _QWORD **v8; // r8
  int v9; // r13d
  unsigned int v10; // edi
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rdi
  unsigned int v15; // eax
  int v16; // esi
  int v17; // edi
  __int64 v18; // rdi
  _QWORD *v19; // rdi
  __int64 v20; // rbx
  __int64 *v21; // rax
  __int64 v22; // rdi
  __int64 v23; // rsi
  _QWORD *v24; // rdi
  __int64 v25; // rbx
  __int64 *v26; // rax
  _BYTE *v27; // [rsp+8h] [rbp-48h]
  char v28; // [rsp+8h] [rbp-48h]
  __int64 v29[2]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v30; // [rsp+20h] [rbp-30h]

  v5 = *(_BYTE **)(a2 - 24);
  v6 = *(_QWORD **)(a3 - 48);
  v7 = *(_QWORD ***)(a2 - 48);
  v8 = *(_QWORD ***)(a3 - 24);
  v9 = *(_WORD *)(a2 + 18) & 0x7FFF;
  v10 = *(_WORD *)(a3 + 18) & 0x7FFF;
  if ( v6 == (_QWORD *)v5 && v8 == v7 )
  {
    v28 = a4;
    v15 = sub_15FF5D0(v10);
    LOBYTE(a4) = v28;
    v5 = v6;
    v10 = v15;
    goto LABEL_17;
  }
  if ( v6 != v7 || v8 != (_QWORD **)v5 )
  {
    if ( v9 == 7 && v10 == 7 )
    {
      if ( (_BYTE)a4 )
        goto LABEL_7;
    }
    else
    {
      LOBYTE(a3) = v9 == 8;
      if ( v10 == 8 && v9 == 8 && !(_BYTE)a4 )
      {
LABEL_7:
        v27 = v8;
        if ( (_QWORD *)*v6 == *v7
          && (unsigned __int8)sub_17272A0(v5, a2, a3, a4)
          && (unsigned __int8)sub_17272A0(v27, a2, v11, v12) )
        {
          v13 = *(_QWORD *)(a1 + 8);
          v30 = 257;
          return sub_17290F0(v13, v9, (__int64)v7, (__int64)v6, v29, 0);
        }
      }
    }
    return 0;
  }
LABEL_17:
  v16 = v10 & v9;
  v17 = v9 | v10;
  if ( !(_BYTE)a4 )
    v16 = v17;
  if ( v16 )
  {
    if ( v16 != 15 )
    {
      v18 = *(_QWORD *)(a1 + 8);
      v30 = 257;
      return sub_17290F0(v18, v16, (__int64)v7, (__int64)v5, v29, 0);
    }
    v19 = (_QWORD *)**v7;
    if ( *((_BYTE *)*v7 + 8) == 16 )
    {
      v20 = (*v7)[4];
      v21 = (__int64 *)sub_1643320(v19);
      v22 = (__int64)sub_16463B0(v21, v20);
    }
    else
    {
      v22 = sub_1643320(v19);
    }
    v23 = 1;
  }
  else
  {
    v24 = (_QWORD *)**v7;
    if ( *((_BYTE *)*v7 + 8) == 16 )
    {
      v25 = (*v7)[4];
      v26 = (__int64 *)sub_1643320(v24);
      v22 = (__int64)sub_16463B0(v26, v25);
    }
    else
    {
      v22 = sub_1643320(v24);
    }
    v23 = 0;
  }
  return (unsigned __int8 *)sub_15A0680(v22, v23, 0);
}
