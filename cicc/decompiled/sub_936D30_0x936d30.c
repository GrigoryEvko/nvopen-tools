// Function: sub_936D30
// Address: 0x936d30
//
__int64 __fastcall sub_936D30(_QWORD *a1, _QWORD *a2)
{
  unsigned __int64 *v2; // rax
  unsigned __int64 v3; // rbx
  _QWORD *v4; // r14
  _QWORD *v5; // r15
  unsigned int v6; // eax
  __int64 v7; // r9
  _QWORD *v8; // rax
  unsigned __int8 v9; // cl
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 v13; // [rsp+0h] [rbp-70h]
  __int64 v14; // [rsp+8h] [rbp-68h]
  unsigned __int64 v15; // [rsp+10h] [rbp-60h]
  _QWORD *v16; // [rsp+18h] [rbp-58h]
  _BYTE v17[80]; // [rsp+20h] [rbp-50h] BYREF

  v2 = (unsigned __int64 *)a2[10];
  v3 = a2[9];
  v14 = a2[6];
  v15 = v2[1];
  if ( *v2 )
    sub_9363D0(a1, *v2);
  v4 = (_QWORD *)sub_945CA0(a1, "for.cond", 0, 0);
  v16 = (_QWORD *)sub_945CA0(a1, "for.end", 0, 0);
  v5 = (_QWORD *)sub_945CA0(a1, "for.body", 0, 0);
  v13 = *(_QWORD *)(a1[4] + 368LL);
  if ( !dword_4D04658 )
  {
    sub_941230(v13, *(unsigned int *)a2);
    sub_9415C0(v13, a1 + 6);
  }
  sub_92FEA0((__int64)a1, v4, 0);
  if ( v14 )
    sub_921E00((__int64)a1, v14);
  else
    sub_ACD6D0(a1[5]);
  v6 = sub_92F9D0(v3, 0);
  sub_945D00(a1, v7, v5, v16, v6);
  sub_92FEA0((__int64)a1, v5, 0);
  if ( v3 )
    sub_9363D0(a1, v3);
  if ( v15 )
  {
    sub_92FD10((__int64)a1, (unsigned int *)(v15 + 36));
    sub_91CAC0((_QWORD *)(v15 + 36));
    v8 = (_QWORD *)sub_945CA0(a1, "for.inc", 0, 0);
    sub_92FEA0((__int64)a1, v8, 0);
    v9 = 0;
    if ( (*(_BYTE *)(*(_QWORD *)v15 + 140LL) & 0xFB) == 8 )
      v9 = (sub_8D4C10(*(_QWORD *)v15, dword_4F077C4 != 2) & 2) != 0;
    sub_921F50((__int64)v17, (__int64)a1, (__int64 *)v15, v9);
  }
  v10 = sub_92FD90((__int64)a1, (__int64)v4);
  v11 = v10;
  if ( v10 )
  {
    if ( a2[8] )
      sub_9305A0((__int64)a1, v10, (__int64)a2);
    sub_930810((__int64)a1, v11);
  }
  if ( !dword_4D04658 )
    sub_93FF00(v13, a1 + 6);
  return sub_92FEA0((__int64)a1, v16, 1);
}
