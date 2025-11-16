// Function: sub_30301B0
// Address: 0x30301b0
//
__int64 __fastcall sub_30301B0(__int64 a1)
{
  _QWORD *v2; // rdx
  int v3; // eax
  __int64 v4; // rcx
  int v5; // eax
  __int64 v6; // rcx
  _QWORD *v7; // rax
  __int64 v8; // rcx
  _QWORD *v9; // rax

  if ( *(_DWORD *)(a1 + 24) != 56 )
    return 0;
  v2 = *(_QWORD **)(a1 + 40);
  v3 = *(_DWORD *)(*v2 + 24LL);
  if ( v3 != 11 && v3 != 35 )
    goto LABEL_5;
  v8 = *(_QWORD *)(*v2 + 96LL);
  v9 = *(_QWORD **)(v8 + 24);
  if ( *(_DWORD *)(v8 + 32) > 0x40u )
    v9 = (_QWORD *)*v9;
  if ( v9 == (_QWORD *)1 )
    return v2[5];
LABEL_5:
  v4 = v2[5];
  v5 = *(_DWORD *)(v4 + 24);
  if ( v5 != 11 && v5 != 35 )
    return 0;
  v6 = *(_QWORD *)(v4 + 96);
  v7 = *(_QWORD **)(v6 + 24);
  if ( *(_DWORD *)(v6 + 32) > 0x40u )
    v7 = (_QWORD *)*v7;
  if ( v7 != (_QWORD *)1 )
    return 0;
  return *v2;
}
