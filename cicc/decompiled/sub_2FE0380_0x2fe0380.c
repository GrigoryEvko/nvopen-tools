// Function: sub_2FE0380
// Address: 0x2fe0380
//
__int64 __fastcall sub_2FE0380(__int64 a1, __int64 a2, _BYTE *a3)
{
  __int64 v3; // r15
  __int64 v4; // r14
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // rax
  int v7; // r12d
  __int64 (*v8)(); // rax
  __int64 (*v10)(); // rax
  unsigned __int64 v11; // [rsp+0h] [rbp-40h]

  v3 = *(_QWORD *)(a2 + 24);
  v4 = *(_QWORD *)(*(_QWORD *)(v3 + 32) + 32LL);
  v5 = sub_2EBEE90(v4, *(_DWORD *)(*(_QWORD *)(a2 + 32) + 48LL));
  v6 = sub_2EBEE90(v4, *(_DWORD *)(*(_QWORD *)(a2 + 32) + 88LL));
  v7 = *(unsigned __int16 *)(a2 + 68);
  v11 = v6;
  if ( sub_2FE0330(a1, v7, *(unsigned __int16 *)(v5 + 68)) || !sub_2FE0330(a1, v7, *(unsigned __int16 *)(v11 + 68)) )
  {
    *a3 = 0;
  }
  else
  {
    v5 = v11;
    *a3 = 1;
  }
  if ( sub_2FE0330(a1, v7, *(unsigned __int16 *)(v5 + 68))
    && (v8 = *(__int64 (**)())(*(_QWORD *)a1 + 640LL), v8 != sub_2FDC5C0)
    && (((unsigned __int8 (__fastcall *)(__int64, unsigned __int64, _QWORD))v8)(a1, v5, 0)
     || (v10 = *(__int64 (**)())(*(_QWORD *)a1 + 640LL), v10 != sub_2FDC5C0)
     && ((unsigned __int8 (__fastcall *)(__int64, unsigned __int64, __int64))v10)(a1, v5, 1))
    && (*(unsigned __int8 (__fastcall **)(__int64, unsigned __int64, __int64))(*(_QWORD *)a1 + 656LL))(a1, v5, v3) )
  {
    return sub_2EBEF70(v4, *(_DWORD *)(*(_QWORD *)(v5 + 32) + 8LL));
  }
  else
  {
    return 0;
  }
}
