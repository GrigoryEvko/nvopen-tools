// Function: sub_164F980
// Address: 0x164f980
//
__int64 __fastcall sub_164F980(__int64 a1)
{
  __int64 v1; // rax
  unsigned __int64 v2; // rdx
  unsigned __int8 v3; // cl
  unsigned __int64 v4; // rsi
  unsigned __int64 v5; // rcx
  __int64 v6; // rdx
  _QWORD *v7; // rax
  unsigned __int64 v9; // rdx
  __int64 v10; // rax

  v1 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  v2 = *(_QWORD *)(a1 - 24 * v1);
  v3 = *(_BYTE *)(v2 + 16);
  if ( v3 == 88 )
  {
    v10 = sub_157F120(*(_QWORD *)(v2 + 40));
    v2 = sub_157EBA0(v10);
    v3 = *(_BYTE *)(v2 + 16);
    v1 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  }
  if ( v3 <= 0x17u )
  {
    v4 = 0;
    goto LABEL_6;
  }
  if ( v3 == 78 )
  {
    v9 = v2 | 4;
  }
  else
  {
    v4 = 0;
    if ( v3 != 29 )
    {
LABEL_6:
      v5 = v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF);
      goto LABEL_7;
    }
    v9 = v2 & 0xFFFFFFFFFFFFFFFBLL;
  }
  v4 = v9 & 0xFFFFFFFFFFFFFFF8LL;
  v5 = (v9 & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (*(_DWORD *)((v9 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF);
  if ( (v9 & 4) == 0 )
    goto LABEL_6;
LABEL_7:
  v6 = *(_QWORD *)(a1 + 24 * (2 - v1));
  v7 = *(_QWORD **)(v6 + 24);
  if ( *(_DWORD *)(v6 + 32) > 0x40u )
    v7 = (_QWORD *)*v7;
  return *(_QWORD *)(v5 + 24LL * (unsigned int)v7);
}
