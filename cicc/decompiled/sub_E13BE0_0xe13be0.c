// Function: sub_E13BE0
// Address: 0xe13be0
//
__int64 *__fastcall sub_E13BE0(__int64 a1, __int64 *a2)
{
  __int64 v3; // r13
  _BYTE *v4; // rdi
  char v5; // al
  char v6; // al
  char v7; // al
  char *v8; // rdx
  __int64 v10; // rax
  __int64 v11; // rax

  v3 = *(_QWORD *)(a1 + 16);
  if ( *(_BYTE *)(v3 + 8) == 11 )
  {
    v10 = *(_QWORD *)(v3 + 16);
    if ( *(_BYTE *)(v10 + 8) == 8 && *(_QWORD *)(v10 + 16) == 11 )
    {
      v11 = *(_QWORD *)(v10 + 24);
      if ( *(_QWORD *)v11 == 0x6A626F5F636A626FLL && *(_WORD *)(v11 + 8) == 25445 && *(_BYTE *)(v11 + 10) == 116 )
      {
        sub_E12F20(a2, 3u, "id<");
        sub_E12F20(a2, *(_QWORD *)(v3 + 24), *(const void **)(v3 + 32));
        v8 = ">";
        return sub_E12F20(a2, 1u, v8);
      }
    }
  }
  (*(void (__fastcall **)(_QWORD, __int64 *))(*(_QWORD *)v3 + 32LL))(*(_QWORD *)(a1 + 16), a2);
  v4 = *(_BYTE **)(a1 + 16);
  v5 = v4[10] & 3;
  if ( v5 != 2 )
  {
    if ( v5 )
      goto LABEL_7;
LABEL_12:
    sub_E12F20(a2, 1u, " ");
    v4 = *(_BYTE **)(a1 + 16);
    v6 = v4[10] & 3;
    if ( v6 == 2 )
      goto LABEL_5;
LABEL_13:
    if ( !v6 )
      goto LABEL_14;
    goto LABEL_7;
  }
  if ( (*(unsigned __int8 (__fastcall **)(_BYTE *, __int64 *))(*(_QWORD *)v4 + 8LL))(v4, a2) )
    goto LABEL_12;
  v4 = *(_BYTE **)(a1 + 16);
  v6 = v4[10] & 3;
  if ( v6 != 2 )
    goto LABEL_13;
LABEL_5:
  if ( (*(unsigned __int8 (__fastcall **)(_BYTE *, __int64 *))(*(_QWORD *)v4 + 8LL))(v4, a2) )
    goto LABEL_14;
  v4 = *(_BYTE **)(a1 + 16);
LABEL_7:
  v7 = v4[10] & 0xC;
  if ( v7 == 8 )
  {
    if ( !(*(unsigned __int8 (__fastcall **)(_BYTE *, __int64 *))(*(_QWORD *)v4 + 16LL))(v4, a2) )
      goto LABEL_9;
    goto LABEL_14;
  }
  if ( !v7 )
LABEL_14:
    sub_E12F20(a2, 1u, "(");
LABEL_9:
  v8 = "*";
  return sub_E12F20(a2, 1u, v8);
}
