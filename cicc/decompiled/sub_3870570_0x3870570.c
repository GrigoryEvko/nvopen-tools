// Function: sub_3870570
// Address: 0x3870570
//
__int64 __fastcall sub_3870570(__int64 a1, __int64 *a2, __int64 a3, char a4)
{
  int v6; // edx
  __int64 *v7; // rbx
  __int64 result; // rax
  __int64 *v9; // rax
  __int64 v10; // rsi
  __int64 v12; // rcx
  __int64 v13; // r15
  __int64 *v14; // rcx
  __int64 v15; // rsi
  unsigned __int8 v16; // al
  char v17; // al
  __int64 v18; // r14
  unsigned int v19; // r12d
  _QWORD *v20; // rax
  int v21; // r12d
  __int64 v22; // r14
  _QWORD *v23; // rax
  __int64 *v24; // [rsp-40h] [rbp-40h]

  if ( a2 == (__int64 *)a3 )
    return 0;
  v6 = *((unsigned __int8 *)a2 + 16);
  if ( v6 == 56 )
  {
    v12 = 24LL * (*((_DWORD *)a2 + 5) & 0xFFFFFFF);
    if ( (*((_BYTE *)a2 + 23) & 0x40) != 0 )
    {
      v9 = (__int64 *)*(a2 - 1);
      v13 = (__int64)(v9 + 3);
      v14 = &v9[(unsigned __int64)v12 / 8];
      if ( v9 + 3 != v14 )
      {
        v15 = *(_QWORD *)v13;
        v16 = *(_BYTE *)(*(_QWORD *)v13 + 16LL);
        if ( v16 > 0x10u )
          goto LABEL_22;
        goto LABEL_31;
      }
      goto LABEL_16;
    }
    v9 = &a2[v12 / 0xFFFFFFFFFFFFFFF8LL];
    v13 = (__int64)&a2[v12 / 0xFFFFFFFFFFFFFFF8LL + 3];
    if ( a2 != (__int64 *)v13 )
    {
      v14 = a2;
      while ( 1 )
      {
        v15 = *(_QWORD *)v13;
        v16 = *(_BYTE *)(*(_QWORD *)v13 + 16LL);
        if ( v16 > 0x10u )
        {
LABEL_22:
          if ( v16 > 0x17u )
          {
            v24 = v14;
            v17 = sub_15CCEE0(*(_QWORD *)(*(_QWORD *)a1 + 56LL), v15, a3);
            v14 = v24;
            if ( !v17 )
              return 0;
          }
          if ( !a4 )
          {
            if ( (*((_DWORD *)a2 + 5) & 0xFFFFFFF) != 2 )
              return 0;
            v18 = *a2;
            v19 = *(_DWORD *)(*a2 + 8);
            v20 = (_QWORD *)sub_15E0530(*(_QWORD *)(*(_QWORD *)a1 + 24LL));
            v21 = v19 >> 8;
            if ( v18 != sub_16471A0(v20, v21) )
            {
              v22 = *a2;
              v23 = (_QWORD *)sub_15E0530(*(_QWORD *)(*(_QWORD *)a1 + 24LL));
              if ( v22 != sub_16471D0(v23, v21) )
                return 0;
            }
LABEL_14:
            if ( (*((_BYTE *)a2 + 23) & 0x40) != 0 )
              v9 = (__int64 *)*(a2 - 1);
            else
              v9 = &a2[-3 * (*((_DWORD *)a2 + 5) & 0xFFFFFFF)];
            break;
          }
        }
LABEL_31:
        v13 += 24;
        if ( (__int64 *)v13 == v14 )
          goto LABEL_14;
      }
    }
LABEL_16:
    result = *v9;
    if ( *(_BYTE *)(result + 16) <= 0x17u )
      return 0;
  }
  else
  {
    if ( (unsigned int)(v6 - 24) <= 0x20 )
    {
      if ( (((_BYTE)v6 - 35) & 0xFD) != 0 )
        return 0;
      if ( (*((_BYTE *)a2 + 23) & 0x40) != 0 )
      {
        v9 = (__int64 *)*(a2 - 1);
        v10 = v9[3];
        if ( *(_BYTE *)(v10 + 16) <= 0x17u )
          goto LABEL_16;
      }
      else
      {
        v9 = &a2[-3 * (*((_DWORD *)a2 + 5) & 0xFFFFFFF)];
        v10 = v9[3];
        if ( *(_BYTE *)(v10 + 16) <= 0x17u )
          goto LABEL_16;
      }
      if ( !sub_15CCEE0(*(_QWORD *)(*(_QWORD *)a1 + 56LL), v10, a3) )
        return 0;
      goto LABEL_14;
    }
    if ( v6 != 71 )
      return 0;
    v7 = (*((_BYTE *)a2 + 23) & 0x40) != 0 ? (__int64 *)*(a2 - 1) : &a2[-3 * (*((_DWORD *)a2 + 5) & 0xFFFFFFF)];
    result = *v7;
    if ( *(_BYTE *)(*v7 + 16) <= 0x17u )
      return 0;
  }
  return result;
}
