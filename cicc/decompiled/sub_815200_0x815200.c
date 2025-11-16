// Function: sub_815200
// Address: 0x815200
//
__int64 __fastcall sub_815200(__int64 a1, _QWORD *a2, _QWORD *a3)
{
  _QWORD *v4; // rbx
  char v5; // al
  _QWORD *v6; // rax
  __int64 v8; // rax
  char *v9; // rdi
  char s[96]; // [rsp+0h] [rbp-60h] BYREF

  v4 = a2;
  if ( !a2 )
    return sub_814FD0(a1, a3);
  while ( 1 )
  {
    v5 = *((_BYTE *)v4 + 8);
    if ( (v5 & 1) != 0 )
    {
      *a3 += 2LL;
      sub_8238B0(qword_4F18BE0, "ix", 2);
      sub_815200(a1, *v4, a3);
      sprintf(s, "%ld", v4[2]);
      v6 = sub_72BA30(unk_4F06A60);
      return sub_80F9E0(s, (__int64)v6, a3);
    }
    if ( (v5 & 2) == 0 )
      break;
    v4 = (_QWORD *)*v4;
    if ( !v4 )
      return sub_814FD0(a1, a3);
  }
  *a3 += 2LL;
  sub_8238B0(qword_4F18BE0, "dt", 2);
  sub_815200(a1, *v4, a3);
  v8 = v4[2];
  if ( (*(_BYTE *)(v8 + 89) & 8) != 0 )
    v9 = *(char **)(v8 + 24);
  else
    v9 = *(char **)(v8 + 8);
  return sub_80BC40(v9, a3);
}
