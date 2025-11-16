// Function: sub_31DDBD0
// Address: 0x31ddbd0
//
__int64 __fastcall sub_31DDBD0(_QWORD *a1, _DWORD *a2, __int64 a3, unsigned int a4, __int64 a5)
{
  __int64 v6; // rcx
  _QWORD *v8; // r15
  __int64 v9; // rax
  unsigned __int64 v10; // r13
  __int64 (*v11)(); // rax
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned __int8 *v14; // r13
  _QWORD *v15; // r14
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 (*v19)(); // rax
  __int64 v20; // rax
  __int64 v21; // rax

  v6 = (unsigned int)*a2;
  switch ( (int)v6 )
  {
    case 0:
      v15 = (_QWORD *)a1[27];
      v16 = sub_2E309C0(a3, (__int64)a2, (unsigned int)v6, v6, a5);
      v14 = (unsigned __int8 *)sub_E808D0(v16, 0, v15, 0);
      break;
    case 1:
    case 2:
    case 5:
      BUG();
    case 3:
    case 4:
      v8 = (_QWORD *)a1[27];
      if ( (_DWORD)v6 == 3 && *(_BYTE *)(a1[26] + 280LL) )
      {
        v21 = sub_31DD620((__int64)a1, a4, *(_DWORD *)(a3 + 24));
        v14 = (unsigned __int8 *)sub_E808D0(v21, 0, v8, 0);
      }
      else
      {
        v9 = sub_2E309C0(a3, (__int64)a2, (unsigned int)v6, v6, a5);
        v10 = sub_E808D0(v9, 0, v8, 0);
        v11 = *(__int64 (**)())(**(_QWORD **)(a1[29] + 16LL) + 144LL);
        if ( v11 == sub_2C8F680 )
          BUG();
        v12 = v11();
        v13 = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD, _QWORD))(*(_QWORD *)v12 + 1944LL))(
                v12,
                a1[29],
                a4,
                a1[27]);
        v14 = (unsigned __int8 *)sub_E81A00(18, v10, v13, (_QWORD *)a1[27], 0);
      }
      break;
    case 6:
      v19 = *(__int64 (**)())(**(_QWORD **)(a1[29] + 16LL) + 144LL);
      if ( v19 == sub_2C8F680 )
        BUG();
      v20 = v19();
      v14 = (unsigned __int8 *)(*(__int64 (__fastcall **)(__int64, _DWORD *, __int64, _QWORD, _QWORD))(*(_QWORD *)v20 + 1928LL))(
                                 v20,
                                 a2,
                                 a3,
                                 a4,
                                 a1[27]);
      break;
    default:
      v14 = 0;
      break;
  }
  v17 = sub_31DA930((__int64)a1);
  sub_2E79AB0(a2, v17);
  return sub_E9A5B0(a1[28], v14);
}
