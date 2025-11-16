// Function: sub_19927B0
// Address: 0x19927b0
//
__int64 __fastcall sub_19927B0(__int64 a1, __int64 *a2, __int64 *a3)
{
  char v3; // al
  __int64 v4; // r12
  char v5; // al
  unsigned int v6; // r13d
  __int64 *v7; // rax
  __int64 v9; // rax
  unsigned int v10; // eax

  v3 = *((_BYTE *)a2 + 16);
  if ( v3 == 55 )
  {
    v4 = *(_QWORD *)*(a2 - 6);
LABEL_3:
    v5 = *(_BYTE *)(v4 + 8);
    goto LABEL_4;
  }
  v4 = *a2;
  if ( v3 == 54 || v3 == 59 || v3 == 58 )
    goto LABEL_3;
  if ( v3 != 78 || (v9 = *(a2 - 3), *(_BYTE *)(v9 + 16)) || (*(_BYTE *)(v9 + 33) & 0x20) == 0 )
  {
    v5 = *(_BYTE *)(v4 + 8);
    goto LABEL_4;
  }
  v10 = *(_DWORD *)(v9 + 36);
  if ( v10 == 137 )
    goto LABEL_19;
  if ( v10 > 0x89 )
  {
    if ( v10 != 148 )
      goto LABEL_17;
LABEL_19:
    v4 = *a3;
    v5 = *(_BYTE *)(*a3 + 8);
    goto LABEL_4;
  }
  if ( (v10 & 0xFFFFFFFD) != 0x85 )
  {
LABEL_17:
    sub_14A36E0(a1);
    goto LABEL_3;
  }
  v4 = *a3;
  v5 = *(_BYTE *)(*a3 + 8);
LABEL_4:
  if ( v5 == 15 )
  {
    v6 = *(_DWORD *)(v4 + 8);
    v7 = (__int64 *)sub_1644900(*(_QWORD **)v4, 1u);
    return sub_1646BA0(v7, v6 >> 8);
  }
  return v4;
}
