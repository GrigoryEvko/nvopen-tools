// Function: sub_705740
// Address: 0x705740
//
__int64 sub_705740()
{
  _QWORD *v0; // r8
  __int64 v1; // rdx
  __int64 i; // rax
  __int64 v3; // rcx
  __int64 result; // rax

  nmemb = 167;
  qword_4D03AC0 = sub_822B10(2672);
  v0 = (_QWORD *)qword_4D03AC0;
  v1 = qword_4D03AC0;
  for ( i = 1; i != 59; ++i )
  {
    v3 = (__int64)*(&off_4B6DCE0 + i);
    *(_BYTE *)(v1 + 8) = i;
    v1 += 16;
    *(_QWORD *)(v1 - 16) = v3;
  }
  v0[116] = off_4A517E0;
  v0[333] = qword_4A51EA8;
  qmemcpy(
    (void *)((unsigned __int64)(v0 + 117) & 0xFFFFFFFFFFFFFFF8LL),
    (const void *)((char *)&off_4A517E0 - ((char *)v0 - ((unsigned __int64)(v0 + 117) & 0xFFFFFFFFFFFFFFF8LL) + 928)),
    8LL * (((_DWORD)v0 + 928 - (((_DWORD)v0 + 936) & 0xFFFFFFF8) + 1744) >> 3));
  qsort(v0, nmemb, 0x10u, (__compar_fn_t)sub_703AB0);
  result = unk_4D04508;
  if ( unk_4D04508 )
    return sub_8539C0(&unk_4D03AA0);
  return result;
}
