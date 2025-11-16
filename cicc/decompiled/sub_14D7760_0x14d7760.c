// Function: sub_14D7760
// Address: 0x14d7760
//
__int64 __fastcall sub_14D7760(unsigned int a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  int v14; // eax
  __int64 v15; // rdx
  __int64 v16; // rdx
  _QWORD *v17; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  _QWORD *v21; // [rsp+8h] [rbp-38h]
  __int64 v22; // [rsp+8h] [rbp-38h]

LABEL_1:
  while ( *((_BYTE *)a2 + 16) != 5 )
  {
LABEL_12:
    if ( *(_BYTE *)(a3 + 16) != 5 )
      return sub_15A37B0((unsigned __int16)a1, a2, a3, 0);
    a1 = sub_15FF5D0(a1);
    v17 = a2;
    a2 = (_QWORD *)a3;
    a3 = (__int64)v17;
  }
  while ( 1 )
  {
    if ( (unsigned __int8)sub_1593BB0(a3) )
    {
      v14 = *((unsigned __int16 *)a2 + 9);
      if ( v14 == 46 )
      {
        v20 = sub_15A9650(a4, *a2, v10, v11, v12, v13);
        a2 = (_QWORD *)sub_15A4750(a2[-3 * (*((_DWORD *)a2 + 5) & 0xFFFFFFF)], v20, 0);
        a3 = sub_15A06D0(*a2);
        goto LABEL_1;
      }
      if ( v14 == 45
        && *a2 == sub_15A9650(
                    a4,
                    *(_QWORD *)a2[-3 * (*((_DWORD *)a2 + 5) & 0xFFFFFFF)],
                    4LL * (*((_DWORD *)a2 + 5) & 0xFFFFFFF),
                    v11,
                    v12,
                    v13) )
      {
        a2 = (_QWORD *)a2[-3 * (*((_DWORD *)a2 + 5) & 0xFFFFFFF)];
        a3 = sub_15A06D0(*a2);
        goto LABEL_1;
      }
    }
    if ( *(_BYTE *)(a3 + 16) != 5 )
      break;
    v15 = *((unsigned __int16 *)a2 + 9);
    if ( *(_WORD *)(a3 + 18) != (_WORD)v15 )
      break;
    if ( (_DWORD)v15 == 46 )
    {
      v22 = sub_15A9650(a4, *a2, v15, v11, v12, v13);
      a2 = (_QWORD *)sub_15A4750(a2[-3 * (*((_DWORD *)a2 + 5) & 0xFFFFFFF)], v22, 0);
      a3 = sub_15A4750(*(_QWORD *)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF)), v22, 0);
      goto LABEL_1;
    }
    if ( (_DWORD)v15 != 45 )
      break;
    if ( *a2 != sub_15A9650(
                  a4,
                  *(_QWORD *)a2[-3 * (*((_DWORD *)a2 + 5) & 0xFFFFFFF)],
                  4LL * (*((_DWORD *)a2 + 5) & 0xFFFFFFF),
                  v11,
                  v12,
                  v13) )
      break;
    v16 = a2[-3 * (*((_DWORD *)a2 + 5) & 0xFFFFFFF)];
    if ( **(_QWORD **)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF)) != *(_QWORD *)v16 )
      break;
    a2 = (_QWORD *)a2[-3 * (*((_DWORD *)a2 + 5) & 0xFFFFFFF)];
    a3 = *(_QWORD *)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF));
    if ( *(_BYTE *)(v16 + 16) != 5 )
      goto LABEL_12;
  }
  if ( a1 - 32 <= 1 && *((_WORD *)a2 + 9) == 27 && (unsigned __int8)sub_1593BB0(a3) )
  {
    v21 = (_QWORD *)sub_14D7760(a1, a2[-3 * (*((_DWORD *)a2 + 5) & 0xFFFFFFF)], a3, a4, a5);
    v19 = sub_14D7760(a1, a2[3 * (1LL - (*((_DWORD *)a2 + 5) & 0xFFFFFFF))], a3, a4, a5);
    return sub_14D6F90((unsigned int)(a1 != 32) + 26, v21, v19, a4);
  }
  return sub_15A37B0((unsigned __int16)a1, a2, a3, 0);
}
