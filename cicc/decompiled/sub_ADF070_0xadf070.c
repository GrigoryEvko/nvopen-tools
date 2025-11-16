// Function: sub_ADF070
// Address: 0xadf070
//
unsigned __int64 __fastcall sub_ADF070(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdx
  unsigned __int64 v8; // rax

  v7 = a6 + 48;
  v8 = *(_QWORD *)(a6 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a6 + 48 != v8 )
  {
    if ( !v8 )
      BUG();
    a6 = (unsigned int)*(unsigned __int8 *)(v8 - 24) - 30;
    if ( (unsigned int)a6 <= 0xA )
      v7 = v8;
  }
  return sub_ADEDB0(a1, a2, a3, a4, a5, a6, v7, 0);
}
