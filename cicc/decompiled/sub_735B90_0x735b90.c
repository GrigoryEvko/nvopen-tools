// Function: sub_735B90
// Address: 0x735b90
//
_BYTE *__fastcall sub_735B90(int a1, __int64 a2, __int64 *a3)
{
  _BYTE *v4; // r13
  __int64 v5; // rax
  _QWORD *v6; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rdi
  __int64 v15; // rax

  if ( a1 != -1 )
  {
    if ( !a1 )
      goto LABEL_3;
    goto LABEL_26;
  }
  if ( (*(_BYTE *)(a2 + 89) & 4) == 0 )
  {
    if ( dword_4F077C4 != 2 || (*(_BYTE *)(a2 + 88) & 0x70) != 0x30 )
    {
      v5 = *(_QWORD *)(a2 + 40);
      if ( v5 )
      {
        if ( *(_BYTE *)(v5 + 28) == 3 )
        {
          v13 = *(_QWORD *)(v5 + 32);
          if ( v13 )
          {
            v4 = *(_BYTE **)(v13 + 128);
            if ( (*(_BYTE *)(v13 + 124) & 1) != 0 )
              v13 = sub_735B70(v13);
            *a3 = *(_QWORD *)(*(_QWORD *)v13 + 96LL);
            return v4;
          }
        }
      }
    }
    if ( *qword_4D03FD0 )
    {
      if ( *(_QWORD *)a2 )
      {
        v6 = sub_72B7A0((_QWORD *)a2);
        v4 = (_BYTE *)v6[1];
        *a3 = (__int64)(v6 + 3);
        if ( v4 )
          return v4;
      }
    }
LABEL_3:
    v4 = *(_BYTE **)(unk_4D03FF0 + 8LL);
    *a3 = unk_4D03FF0 + 24LL;
    return v4;
  }
  v8 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 32LL);
  if ( !v8 )
  {
LABEL_26:
    v14 = qword_4F04C68[0] + 776LL * a1;
    v4 = sub_732EF0(v14);
    v15 = *(_QWORD *)(v14 + 24);
    if ( !v15 )
      v15 = v14 + 32;
    *a3 = v15;
    return v4;
  }
  v4 = *(_BYTE **)(*(_QWORD *)(v8 + 168) + 152LL);
  if ( !v4 || (v4[29] & 0x20) != 0 || (v9 = *((int *)v4 + 60), (_DWORD)v9 == -1) )
  {
    *a3 = 0;
    return v4;
  }
  v10 = qword_4F04C68[0] + 776 * v9;
  v11 = *(_QWORD *)(v10 + 24);
  v12 = v10 + 32;
  if ( !v11 )
    v11 = v12;
  *a3 = v11;
  return v4;
}
