// Function: sub_2DB4750
// Address: 0x2db4750
//
__int64 __fastcall sub_2DB4750(__int64 **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v6; // rdi
  __int64 (*v7)(); // rax
  __int64 v8; // rbx
  __int64 v9; // r13
  __int16 v10; // ax
  __int64 *v12; // rdi
  __int64 *v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 (__fastcall *v16)(__int64, __int64); // r8
  __int64 (__fastcall *v17)(__int64, __int64); // rax
  __int64 v18; // rax
  __int64 (*v19)(); // rax
  __int64 (*v20)(); // rax

  v6 = *a1;
  v7 = *(__int64 (**)())(*v6 + 200);
  if ( v7 == sub_2DB1AD0
    || !((unsigned __int8 (__fastcall *)(__int64 *, __int64, _QWORD))v7)(v6, a2, (unsigned int)qword_501D0E8)
    || (_BYTE)qword_501D008 )
  {
    v8 = *(_QWORD *)(a2 + 56);
    v9 = sub_2E313E0(a2, a2, a3, a4, a5);
    if ( v8 == v9 )
      return 1;
    while ( 1 )
    {
      v10 = *(_WORD *)(v8 + 68);
      if ( (unsigned __int16)(v10 - 14) > 4u )
      {
        if ( v10 == 68 || !v10 )
          break;
        v12 = *a1;
        v13 = a1[41];
        v14 = *((unsigned int *)a1 + 84);
        v15 = **a1;
        v16 = *(__int64 (__fastcall **)(__int64, __int64))(v15 + 1000);
        if ( v16 == sub_2DB2060 )
        {
          v17 = *(__int64 (__fastcall **)(__int64, __int64))(v15 + 992);
          if ( v17 == sub_2DB1B50 )
            v18 = (*(_QWORD *)(*(_QWORD *)(v8 + 16) + 24LL) >> 22) & 1LL;
          else
            LOBYTE(v18) = ((__int64 (__fastcall *)(__int64 *, __int64, __int64 *, __int64))v17)(v12, v8, v13, v14);
        }
        else
        {
          LOBYTE(v18) = ((__int64 (__fastcall *)(__int64 *, __int64, __int64 *, __int64))v16)(v12, v8, v13, v14);
        }
        if ( !(_BYTE)v18 )
          break;
        v19 = *(__int64 (**)())(**a1 + 920);
        if ( v19 != sub_2DB1B30 )
        {
          if ( ((unsigned __int8 (__fastcall *)(__int64 *, __int64, __int64 *, __int64))v19)(*a1, v8, v13, v14) )
          {
            v20 = *(__int64 (**)())(**a1 + 928);
            if ( v20 == sub_2DB1B40 || !((unsigned __int8 (__fastcall *)(__int64 *, __int64))v20)(*a1, v8) )
              break;
          }
        }
        if ( !(unsigned __int8)sub_2DB4530((__int64)a1, v8) )
          break;
      }
      if ( (*(_BYTE *)v8 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v8 + 44) & 8) != 0 )
          v8 = *(_QWORD *)(v8 + 8);
      }
      v8 = *(_QWORD *)(v8 + 8);
      if ( v8 == v9 )
        return 1;
    }
  }
  return 0;
}
