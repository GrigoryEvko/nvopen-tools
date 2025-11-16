// Function: sub_1689130
// Address: 0x1689130
//
__int64 sub_1689130()
{
  __int64 v0; // r12
  unsigned __int64 i; // r12
  __int64 v2; // rax
  bool v3; // zf
  __int64 result; // rax

  v0 = unk_4F9F820;
  if ( unk_4F9F820 )
  {
    while ( (_UNKNOWN *)v0 != &unk_4F9F600 )
    {
      while ( v0 )
      {
        sub_1688E30();
        if ( *(_BYTE *)(v0 + 272) )
          sub_1688E70();
        else
          sub_1688E80(v0);
        v0 = unk_4F9F820;
        if ( (_UNKNOWN *)unk_4F9F820 == &unk_4F9F600 )
          goto LABEL_7;
      }
    }
LABEL_7:
    for ( i = unk_4F9F5E0; (_UNKNOWN *)unk_4F9F5E0 != &unk_4F9F3C0; i = unk_4F9F5E0 )
    {
      sub_1688E30();
      v2 = *(_QWORD *)(i + 264);
      *(_QWORD *)(*(_QWORD *)(i + 256) + 264LL) = v2;
      *(_QWORD *)(v2 + 256) = *(_QWORD *)(i + 256);
      v3 = *(_BYTE *)(i + 273) == 0;
      *(_QWORD *)(i + 256) = 0;
      *(_QWORD *)(i + 264) = 0;
      if ( v3 )
        _libc_free(i);
      sub_1688E70();
    }
    pthread_mutex_destroy(&stru_4F9F840);
    pthread_key_delete(dword_4F9F868);
    memset(&unk_4F9F720, 0, 0x118u);
    memset(&unk_4F9F600, 0, 0x118u);
    return 0;
  }
  return result;
}
