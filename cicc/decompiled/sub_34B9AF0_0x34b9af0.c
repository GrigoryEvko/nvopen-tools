// Function: sub_34B9AF0
// Address: 0x34b9af0
//
__int64 __fastcall sub_34B9AF0(__int64 a1, _BYTE *a2, unsigned __int8 a3)
{
  __int64 v3; // r14
  _QWORD *v4; // rax
  char v5; // dl
  _QWORD *i; // rbx
  __int64 v7; // r15
  __int64 v8; // rax
  int v9; // eax
  __int64 (*v11)(); // rax
  __int64 v12; // r12
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // rdi
  __int64 (*v17)(); // rax
  __int64 v18; // [rsp+0h] [rbp-40h]

  v3 = *(_QWORD *)(a1 + 40);
  v4 = (_QWORD *)(*(_QWORD *)(v3 + 48) & 0xFFFFFFFFFFFFFFF8LL);
  if ( v4 == (_QWORD *)(v3 + 48)
    || !v4
    || (v18 = (__int64)(v4 - 3), (unsigned int)*((unsigned __int8 *)v4 - 24) - 30 > 0xA) )
  {
    BUG();
  }
  v5 = *((_BYTE *)v4 - 24);
  if ( v5 != 30 )
  {
    if ( (a2[865] & 2) == 0 && ((((*(_WORD *)(a1 + 2) >> 2) & 0x3FF) - 18) & 0xFFFD) != 0 || v5 != 36 )
      return 0;
    v18 = 0;
  }
  for ( i = (_QWORD *)(*v4 & 0xFFFFFFFFFFFFFFF8LL); !i; i = (_QWORD *)(*i & 0xFFFFFFFFFFFFFFF8LL) )
  {
    v7 = 0;
LABEL_7:
    if ( sub_B46AA0(v7) )
      continue;
    if ( *(_BYTE *)v7 == 85
      && (v8 = *(_QWORD *)(v7 - 32)) != 0
      && !*(_BYTE *)v8
      && *(_QWORD *)(v8 + 24) == *(_QWORD *)(v7 + 80)
      && (*(_BYTE *)(v8 + 33) & 0x20) != 0 )
    {
      v9 = *(_DWORD *)(v8 + 36);
      if ( ((v9 - 155) & 0xFFFFFFEF) == 0 || v9 == 11 || v9 == 210 )
        continue;
      if ( (unsigned __int8)sub_B46970((unsigned __int8 *)v7) )
        return 0;
    }
    else if ( (unsigned __int8)sub_B46970((unsigned __int8 *)v7) )
    {
      return 0;
    }
    if ( (unsigned __int8)sub_B46420(v7) || !sub_991A70((unsigned __int8 *)v7, 0, 0, 0, 0, 1u, 0) )
      return 0;
  }
  v7 = (__int64)(i - 3);
  if ( (_QWORD *)a1 != i - 3 )
    goto LABEL_7;
  v11 = *(__int64 (**)())(*(_QWORD *)a2 + 16LL);
  if ( v11 == sub_23CE270 )
    BUG();
  v12 = *(_QWORD *)(v3 + 72);
  v13 = ((__int64 (__fastcall *)(_BYTE *, __int64))v11)(a2, v12);
  v15 = 0;
  v16 = v13;
  v17 = *(__int64 (**)())(*(_QWORD *)v13 + 144LL);
  if ( v17 != sub_2C8F680 )
    v15 = ((__int64 (__fastcall *)(__int64, __int64, __int64, _QWORD))v17)(v16, v12, v14, 0);
  return sub_34B9AB0(v12, v7, v18, v15, a3);
}
