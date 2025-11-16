// Function: sub_2D5C100
// Address: 0x2d5c100
//
__int64 __fastcall sub_2D5C100(__int64 *a1, unsigned __int8 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  int v6; // edx
  __int64 (*v7)(); // r15
  unsigned __int8 *v8; // rdx
  unsigned int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // rbx
  unsigned int v12; // r14d
  __int64 v13; // rcx
  __int64 v14; // r8
  unsigned int v15; // eax
  __int64 v16; // rdx
  __int64 result; // rax
  __int64 v18; // rax
  __int64 (*v19)(); // rcx
  unsigned __int8 *v20; // rsi
  __int64 (*v21)(); // rdx

  v6 = *a2;
  if ( v6 != 69 )
  {
    if ( v6 == 75 )
    {
      v7 = *(__int64 (**)())(*a1 + 1560);
      if ( (a2[7] & 0x40) != 0 )
        v8 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
      else
        v8 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v9 = sub_30097B0(*(_QWORD *)(*(_QWORD *)v8 + 8LL), 0, v8, a4, a5);
      v11 = v10;
      v12 = v9;
      v15 = sub_30097B0(*((_QWORD *)a2 + 1), 0, v10, v13, v14);
      if ( v7 != sub_2D566A0
        && ((unsigned __int8 (__fastcall *)(__int64 *, _QWORD, __int64, _QWORD, __int64))v7)(a1, v15, v16, v12, v11) )
      {
        return 1;
      }
    }
    else
    {
      if ( v6 != 68 )
        BUG();
      v18 = *a1;
      v19 = *(__int64 (**)())(*a1 + 1424);
      if ( (a2[7] & 0x40) != 0 )
        v20 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
      else
        v20 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      if ( v19 == sub_2D56670 )
        goto LABEL_14;
      if ( ((unsigned __int8 (__fastcall *)(__int64 *, _QWORD, _QWORD))v19)(
             a1,
             *(_QWORD *)(*(_QWORD *)v20 + 8LL),
             *((_QWORD *)a2 + 1)) )
      {
        return 1;
      }
    }
  }
  v18 = *a1;
LABEL_14:
  v21 = *(__int64 (**)())(v18 + 1816);
  result = 0;
  if ( v21 != sub_2D566C0 )
    return ((__int64 (__fastcall *)(__int64 *, unsigned __int8 *))v21)(a1, a2);
  return result;
}
