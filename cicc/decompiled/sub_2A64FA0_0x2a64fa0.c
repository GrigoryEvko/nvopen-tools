// Function: sub_2A64FA0
// Address: 0x2a64fa0
//
__int64 __fastcall sub_2A64FA0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  _BYTE *v6; // rax
  char v7; // r14
  _BYTE *v8; // rbx
  unsigned int v9; // eax
  unsigned int v10; // esi
  unsigned int v11; // eax
  const void **v12; // r12
  unsigned int v13; // eax
  unsigned int v14; // eax

  if ( *(_BYTE *)a3 <= 0x15u )
  {
    sub_AD8380(a1, a3);
    return a1;
  }
  if ( !(unsigned __int8)sub_B19060(a2[1], a3, a3, a4) )
  {
    v6 = (_BYTE *)sub_2A64F10(*a2, a3);
    v7 = *v6;
    v8 = v6;
    if ( *v6 == 4 )
    {
      v12 = (const void **)(v6 + 8);
    }
    else
    {
      v9 = sub_BCB060(*(_QWORD *)(a3 + 8));
      v10 = v9;
      if ( v7 != 5 )
        goto LABEL_7;
      v12 = (const void **)(v8 + 8);
      v10 = v9;
      if ( !sub_9876C0((__int64 *)v8 + 1) )
      {
        v7 = *v8;
LABEL_7:
        if ( v7 == 2 )
        {
          sub_AD8380(a1, *((_QWORD *)v8 + 1));
        }
        else if ( v7 )
        {
          sub_AADB10(a1, v10, 1);
        }
        else
        {
          sub_AADB10(a1, v10, 0);
        }
        return a1;
      }
    }
    v13 = *((_DWORD *)v8 + 4);
    *(_DWORD *)(a1 + 8) = v13;
    if ( v13 > 0x40 )
      sub_C43780(a1, v12);
    else
      *(_QWORD *)a1 = *((_QWORD *)v8 + 1);
    v14 = *((_DWORD *)v8 + 8);
    *(_DWORD *)(a1 + 24) = v14;
    if ( v14 > 0x40 )
      sub_C43780(a1 + 16, (const void **)v8 + 3);
    else
      *(_QWORD *)(a1 + 16) = *((_QWORD *)v8 + 3);
    return a1;
  }
  v11 = sub_BCB060(*(_QWORD *)(a3 + 8));
  sub_AADB10(a1, v11, 1);
  return a1;
}
