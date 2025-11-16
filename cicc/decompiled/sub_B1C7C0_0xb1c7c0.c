// Function: sub_B1C7C0
// Address: 0xb1c7c0
//
__int64 __fastcall sub_B1C7C0(__int64 a1, _QWORD *a2)
{
  _QWORD *v2; // rdx
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rbx
  int v5; // eax
  unsigned __int64 v6; // rdx

  v2 = (_QWORD *)(*a2 + 48LL);
  v3 = *v2 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v3 == v2 )
    goto LABEL_6;
  if ( !v3 )
    BUG();
  v4 = v3 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v3 - 24) - 30 > 0xA )
  {
LABEL_6:
    v5 = 0;
    v6 = 0;
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
