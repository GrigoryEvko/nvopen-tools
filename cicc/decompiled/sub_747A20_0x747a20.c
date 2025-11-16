// Function: sub_747A20
// Address: 0x747a20
//
void __fastcall sub_747A20(__int64 a1, __int64 a2)
{
  char v3; // al
  __int64 v4; // r13
  int v5; // r14d
  char v6; // r15
  __int64 v7; // rax
  __int64 v8; // rdi
  void (__fastcall *v9)(__int64, _QWORD); // rax
  void (__fastcall *v10)(__int64, __int64); // rax
  __int64 v11; // rdi

  v3 = *(_BYTE *)(a1 + 176);
  if ( v3 != 11 )
  {
    if ( v3 == 3 )
    {
      v5 = 0;
      v4 = a1;
      if ( *(_QWORD *)(a1 + 184) )
      {
LABEL_10:
        v7 = *(_QWORD *)(v4 + 40);
        if ( v7 )
        {
          if ( *(_BYTE *)(a2 + 136) && unk_4F068C0 && (*(_BYTE *)(a1 + 177) & 1) == 0 )
          {
            *(_BYTE *)(a2 + 160) = 1;
          }
          else
          {
            v8 = *(_QWORD *)(v7 + 32);
            v9 = *(void (__fastcall **)(__int64, _QWORD))(a2 + 40);
            if ( v9 )
              v9(v8, 0);
            else
              sub_74C3E0(v8, a2);
          }
        }
        else
        {
          (*(void (__fastcall **)(const char *, __int64))a2)("<null parent scope>::", a2);
        }
        (*(void (__fastcall **)(const char *, __int64))a2)("operator ", a2);
        sub_74B930(*(_QWORD *)(v4 + 184), a2);
        *(_BYTE *)(a2 + 160) = 0;
LABEL_7:
        if ( !v5 )
          return;
        goto LABEL_19;
      }
    }
    v4 = a1;
    v5 = 0;
    v6 = *(_BYTE *)(a2 + 142);
    *(_BYTE *)(a2 + 142) = *(_BYTE *)(a1 + 177) & 1;
LABEL_4:
    if ( (*(_BYTE *)(v4 + 90) & 0x30) == 0x10 )
      sub_74C010(v4, 2, a2);
    else
      sub_74C550(v4, 2, a2);
    *(_BYTE *)(a2 + 142) = v6;
    goto LABEL_7;
  }
  v4 = *(_QWORD *)(a1 + 184);
  if ( *(_BYTE *)(v4 + 176) == 3 )
  {
    v5 = 1;
    if ( *(_QWORD *)(v4 + 184) )
      goto LABEL_10;
  }
  v6 = *(_BYTE *)(a2 + 142);
  *(_BYTE *)(a2 + 142) = *(_BYTE *)(v4 + 177) & 1;
  v10 = *(void (__fastcall **)(__int64, __int64))(a2 + 32);
  if ( !v10 )
  {
    v5 = 1;
    goto LABEL_4;
  }
  v10(v4, 2);
  *(_BYTE *)(a2 + 142) = v6;
LABEL_19:
  v11 = *(_QWORD *)(a1 + 192);
  if ( v11 )
    sub_7477E0(v11, 0, a2);
  else
    (*(void (__fastcall **)(const char *, __int64))a2)("<>", a2);
}
