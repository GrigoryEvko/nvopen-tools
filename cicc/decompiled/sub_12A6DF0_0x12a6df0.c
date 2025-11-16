// Function: sub_12A6DF0
// Address: 0x12a6df0
//
__int64 __fastcall sub_12A6DF0(__int64 a1, __int64 a2, int a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 v5; // r14
  __int64 v6; // r13
  __int64 v7; // r15
  char *v8; // r14
  char *v9; // rax
  int v10; // eax
  int v11; // r13d
  unsigned int v12; // eax
  __int64 v14; // [rsp+10h] [rbp-50h]
  _DWORD *v15; // [rsp+18h] [rbp-48h]
  int v16; // [rsp+18h] [rbp-48h]
  __int64 v17[7]; // [rsp+28h] [rbp-38h] BYREF

  v4 = *(_QWORD *)(a4 + 72);
  v17[0] = 1;
  v5 = *(_QWORD *)(v4 + 16);
  v15 = (_DWORD *)(a4 + 36);
  v6 = *(_QWORD *)(v5 + 16);
  v7 = *(_QWORD *)(v6 + 16);
  if ( a3 == 410 && !sub_127C7B0(*(_QWORD *)(v7 + 16), v17) )
    sub_127B550("align value for memset was not constant", v15, 1);
  v8 = sub_128F980(a2, v5);
  v14 = sub_1643330(*(_QWORD *)(a2 + 40));
  v9 = sub_128F980(a2, v6);
  v10 = sub_128B420(a2, v9, 0, v14, 0, 0, v15);
  v11 = v17[0];
  v16 = v10;
  v12 = (unsigned int)sub_128F980(a2, v7);
  sub_15E7280(a2 + 48, (_DWORD)v8, v16, v12, v11, 0, 0, 0, 0);
  *(_QWORD *)a1 = v8;
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  return a1;
}
