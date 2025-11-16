// Function: sub_2B2D0F0
// Address: 0x2b2d0f0
//
unsigned __int64 __fastcall sub_2B2D0F0(__int64 *a1, __int64 a2)
{
  int v4; // r12d
  __int64 v5; // rdi
  int v6; // eax
  __int64 *v7; // r15
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // r10
  unsigned __int64 v11; // r13
  _BYTE *v12; // rax
  __int64 v13; // rdi
  int v14; // eax
  int v15; // esi
  __int64 v16; // rax
  unsigned int v17; // r8d
  __int64 v18; // r11
  __int64 v19; // rdx
  unsigned int v20; // eax
  __int64 v21; // rax
  unsigned __int64 result; // rax
  __int64 v23; // [rsp+8h] [rbp-98h]
  __int64 v24; // [rsp+10h] [rbp-90h]
  __int64 v25; // [rsp+10h] [rbp-90h]
  char *v26; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v27; // [rsp+28h] [rbp-78h]
  char v28; // [rsp+30h] [rbp-70h] BYREF

  v4 = *(_QWORD *)(a1[1] + 8);
  v5 = sub_BCB2A0(*(_QWORD **)(*a1 + 3440));
  v6 = *(unsigned __int8 *)(v5 + 8);
  if ( (_BYTE)v6 == 17 )
  {
    v4 *= *(_DWORD *)(v5 + 32);
LABEL_3:
    v5 = **(_QWORD **)(v5 + 16);
    goto LABEL_4;
  }
  if ( (unsigned int)(v6 - 17) <= 1 )
    goto LABEL_3;
LABEL_4:
  sub_BCDA70((__int64 *)v5, v4);
  v7 = *(__int64 **)(*a1 + 3296);
  v8 = *(_QWORD *)(*(_QWORD *)a1[2] + 240LL);
  sub_2B2BBE0(*a1, *(char **)(v8 + 80), *(unsigned int *)(v8 + 88));
  v9 = *(_QWORD *)(*(_QWORD *)a1[2] + 240LL);
  sub_2B2BBE0(*a1, *(char **)v9, *(unsigned int *)(v9 + 8));
  v10 = sub_DFD2D0(v7, (unsigned int)**(unsigned __int8 **)(*(_QWORD *)a1[2] + 416LL) - 29, *(_QWORD *)a1[3]);
  v11 = v10;
  v12 = *(_BYTE **)a1[6];
  if ( *v12 != 86 )
    goto LABEL_15;
  v13 = *(_QWORD *)(*((_QWORD *)v12 - 12) + 8LL);
  v14 = *(unsigned __int8 *)(v13 + 8);
  v15 = *(_QWORD *)(a1[1] + 8);
  if ( (_BYTE)v14 == 17 )
  {
    v15 *= *(_DWORD *)(v13 + 32);
  }
  else if ( (unsigned int)(v14 - 17) > 1 )
  {
    goto LABEL_8;
  }
  v13 = **(_QWORD **)(v13 + 16);
LABEL_8:
  v24 = v10;
  v16 = sub_BCDA70((__int64 *)v13, v15);
  v17 = *(_DWORD *)(v16 + 32);
  v18 = v16;
  v19 = *(_QWORD *)a1[3];
  v20 = 1;
  if ( *(_BYTE *)(v19 + 8) == 17 )
    v20 = *(_DWORD *)(v19 + 32);
  v23 = v24;
  v25 = v18;
  if ( v17 != v20 )
  {
    sub_9B9470((__int64)&v26, v20 / v17, v17);
    v21 = sub_DFBC30(*(__int64 **)(*a1 + 3296), 7, v25, (__int64)v26, v27, 0, 0, 0, 0, 0, 0);
    if ( __OFADD__(v21, v23) )
    {
      v11 = 0x7FFFFFFFFFFFFFFFLL;
      if ( v21 <= 0 )
        v11 = 0x8000000000000000LL;
    }
    else
    {
      v11 = v21 + v23;
    }
    if ( v26 != &v28 )
      _libc_free((unsigned __int64)v26);
  }
LABEL_15:
  result = a2 + v11;
  if ( __OFADD__(a2, v11) )
  {
    result = 0x7FFFFFFFFFFFFFFFLL;
    if ( a2 <= 0 )
      return 0x8000000000000000LL;
  }
  return result;
}
