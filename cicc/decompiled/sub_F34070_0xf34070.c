// Function: sub_F34070
// Address: 0xf34070
//
__int64 __fastcall sub_F34070(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rsi
  unsigned __int64 v3; // rax
  __int64 v4; // rbx
  int v5; // eax
  __int64 v6; // rdx

  v2 = (_QWORD *)(a2 + 48);
  v3 = *v2 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v3 == v2 )
    goto LABEL_6;
  if ( !v3 )
    BUG();
  v4 = v3 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v3 - 24) - 30 > 0xA )
  {
LABEL_6:
    v6 = 0;
    v5 = 0;
    v4 = 0;
  }
  else
  {
    v5 = sub_B46E30(v4);
    v6 = v4;
  }
  *(_QWORD *)a1 = v4;
  *(_DWORD *)(a1 + 24) = v5;
  *(_DWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = v6;
  return a1;
}
