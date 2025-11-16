// Function: sub_7239A0
// Address: 0x7239a0
//
_DWORD *sub_7239A0()
{
  __int64 v0; // rdx
  int v1; // esi
  __int64 v2; // rdx
  const char *v3; // r12
  size_t v4; // rax
  char *v5; // rax
  void (*v7)(int); // rax

  if ( dword_4B6EB20 )
  {
    qword_4F07900 = signal(2, (__sighandler_t)1);
    if ( qword_4F07900 != (__sighandler_t)1 )
      signal(2, (__sighandler_t)handler);
    qword_4F078F8 = signal(15, (__sighandler_t)handler);
    v7 = signal(25, (__sighandler_t)1);
    dword_4B6EB20 = 0;
    qword_4F078F0 = v7;
    dword_4F07908 = 1;
  }
  qword_4F07920 = 0;
  unk_4F06C40 = 0;
  qword_4F06C50 = 0;
  unk_4F06C38 = 0;
  qword_4F06C48 = 0;
  unk_4F076D8 = 0;
  qword_4F07938 = 0;
  unk_4F076D0 = 0;
  qword_4F07940 = 0;
  qword_4F076C8 = 0;
  dword_4F07950 = 0;
  qword_4F076C0 = 0;
  unk_4F07678 = 0;
  dword_4F077C4 = 2;
  qword_4F07960 = newlocale(2, "C", 0);
  dataset = uselocale(qword_4F07960);
  if ( !dataset || !qword_4F07960 )
    sub_721090();
  v1 = qword_4F06C48;
  if ( qword_4F06C48 <= 0xFFu )
  {
    sub_729510(256, qword_4F06C48, v0);
    v1 = qword_4F06C48;
  }
  while ( !getcwd((char *)qword_4F06C50, v1) && *__errno_location() == 34 )
  {
    v1 = qword_4F06C48;
    if ( qword_4F06C48 <= 0xFFFFFFFFFFFFFEFFLL )
    {
      sub_729510(qword_4F06C48 + 256LL, qword_4F06C48, v2);
      v1 = qword_4F06C48;
    }
  }
  v3 = (const char *)sub_721290((__int64)qword_4F06C50);
  v4 = strlen(v3);
  qword_4F076B0 = (char *)sub_822B10(v4 + 1);
  strcpy(qword_4F076B0, v3);
  v5 = getenv("EDG_BASE");
  if ( v5 )
    qword_4F07578 = v5;
  else
    qword_4F07578 = (char *)byte_3F871B3;
  qword_4F07948 = 0;
  qword_4F07930 = 0;
  unk_4F07580 = 1;
  qword_4F07928 = 0;
  qword_4F076F0 = 0;
  qword_4F07918 = 0;
  unk_4F076E8 = 0;
  qword_4F07910 = 0;
  unk_4F076E0 = 0;
  qword_4F078E0 = 0;
  unk_4F076B8 = 0;
  qword_4F078C8 = 0;
  qword_4F076A8 = 0;
  qword_4F078C0 = 0;
  qword_4F076A0 = 0;
  unk_4F07688 = 0;
  dword_4F07680[0] = 1;
  unk_4F0759C = 0;
  dword_4F07598 = 0;
  dword_4F07594 = 0;
  dword_4F078D0 = 0;
  qword_4F07698 = 0;
  qword_4F078D8 = 0;
  qword_4F07690 = 0;
  fd = 0;
  dword_4F07590 = 1;
  unk_4F0758C = 1;
  dword_4F07588 = 0;
  unk_4F07584 = 1;
  unk_4F07574 = 0;
  unk_4F07540 = 0;
  unk_4F07550 = 0;
  unk_4F07560 = 0;
  unk_4F076F8 = 1;
  unk_4F076FC = 1;
  unk_4F07668 = 0;
  dword_4F07518 = 1;
  return &dword_4F07518;
}
