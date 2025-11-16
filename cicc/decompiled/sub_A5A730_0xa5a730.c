// Function: sub_A5A730
// Address: 0xa5a730
//
_BYTE *__fastcall sub_A5A730(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int8 v5; // al
  __int64 v6; // rdi
  char v7; // r12
  int v8; // ebx
  _BYTE *v10; // rax
  unsigned __int8 v11; // si
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // r14
  __int64 (__fastcall *v15)(__int64, __int64); // rax
  __int64 v16; // rdi
  int v17; // eax
  __int64 v18; // rax
  __int64 v19; // r14
  __int64 (__fastcall *v20)(__int64, __int64); // rax

  if ( (*(_BYTE *)(a2 + 7) & 0x10) != 0 )
    return sub_A55040(a1, (_BYTE *)a2);
  v5 = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 <= 0x15u )
  {
    if ( v5 > 3u )
      return (_BYTE *)sub_A5AA10();
    v6 = *(_QWORD *)(a3 + 16);
    if ( v6 )
    {
      v7 = 64;
      v8 = sub_A5A4B0(v6, a2);
      goto LABEL_6;
    }
    goto LABEL_27;
  }
  if ( v5 != 25 )
  {
    if ( v5 == 24 )
      return (_BYTE *)sub_A5C090(a1, *(_QWORD *)(a2 + 24));
    v16 = *(_QWORD *)(a3 + 16);
    if ( v16 )
    {
      v17 = sub_A5A650(v16, a2);
      v11 = 37;
      v8 = v17;
      if ( v17 != -1 )
      {
LABEL_26:
        v12 = sub_A51310(a1, v11);
        return (_BYTE *)sub_CB59F0(v12, v8);
      }
      v18 = sub_A55D90(a2);
      v19 = v18;
      if ( !v18 )
        return (_BYTE *)sub_904010(a1, "<badref>");
      v8 = sub_A5A650(v18, a2);
      v20 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v19 + 8LL);
      if ( v20 == sub_A554F0 )
      {
        sub_A552A0(v19, a2);
        v7 = 37;
        j_j___libc_free_0(v19, 400);
      }
      else
      {
        ((void (__fastcall *)(__int64))v20)(v19);
        v7 = 37;
      }
LABEL_6:
      if ( v8 == -1 )
        return (_BYTE *)sub_904010(a1, "<badref>");
      v11 = v7;
      goto LABEL_26;
    }
LABEL_27:
    v13 = sub_A55D90(a2);
    v14 = v13;
    if ( !v13 )
      return (_BYTE *)sub_904010(a1, "<badref>");
    if ( *(_BYTE *)a2 > 3u )
    {
      v7 = 37;
      v8 = sub_A5A650(v13, a2);
    }
    else
    {
      v7 = 64;
      v8 = sub_A5A4B0(v13, a2);
    }
    v15 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v14 + 8LL);
    if ( v15 == sub_A554F0 )
    {
      sub_A552A0(v14, a2);
      j_j___libc_free_0(v14, 400);
    }
    else
    {
      ((void (__fastcall *)(__int64))v15)(v14);
    }
    goto LABEL_6;
  }
  sub_904010(a1, "asm ");
  if ( *(_BYTE *)(a2 + 96) )
    sub_904010(a1, "sideeffect ");
  if ( *(_BYTE *)(a2 + 97) )
    sub_904010(a1, "alignstack ");
  if ( *(_DWORD *)(a2 + 100) == 1 )
    sub_904010(a1, "inteldialect ");
  if ( *(_BYTE *)(a2 + 104) )
    sub_904010(a1, "unwind ");
  v10 = *(_BYTE **)(a1 + 32);
  if ( (unsigned __int64)v10 >= *(_QWORD *)(a1 + 24) )
  {
    sub_CB5D20(a1, 34);
  }
  else
  {
    *(_QWORD *)(a1 + 32) = v10 + 1;
    *v10 = 34;
  }
  sub_C92400(*(_QWORD *)(a2 + 24), *(_QWORD *)(a2 + 32), a1);
  sub_904010(a1, "\", \"");
  sub_C92400(*(_QWORD *)(a2 + 56), *(_QWORD *)(a2 + 64), a1);
  return (_BYTE *)sub_A51310(a1, 0x22u);
}
