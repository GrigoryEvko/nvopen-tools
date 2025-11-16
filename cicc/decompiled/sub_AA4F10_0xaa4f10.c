// Function: sub_AA4F10
// Address: 0xaa4f10
//
unsigned __int64 __fastcall sub_AA4F10(__int64 a1)
{
  _QWORD *v1; // rax
  _QWORD *v2; // rdx
  unsigned __int64 v3; // rax
  __int64 v4; // rdx

  v1 = (_QWORD *)(*(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL);
  if ( v1 == (_QWORD *)(a1 + 48) )
    return 0;
  if ( !v1 )
    BUG();
  if ( *((_BYTE *)v1 - 24) == 30
    && ((v2 = *(_QWORD **)(a1 + 56)) == 0 || v1 != v2)
    && *(_QWORD **)(v1[2] + 56LL) != v1
    && (v3 = *v1 & 0xFFFFFFFFFFFFFFF8LL) != 0
    && *(_BYTE *)(v3 - 24) == 85
    && (v4 = *(_QWORD *)(v3 - 56)) != 0
    && !*(_BYTE *)v4
    && *(_QWORD *)(v4 + 24) == *(_QWORD *)(v3 + 56)
    && *(_DWORD *)(v4 + 36) == 146 )
  {
    return v3 - 24;
  }
  else
  {
    return 0;
  }
}
