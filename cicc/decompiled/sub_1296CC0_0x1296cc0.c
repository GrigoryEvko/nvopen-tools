// Function: sub_1296CC0
// Address: 0x1296cc0
//
__int64 __fastcall sub_1296CC0(__int64 *a1, _QWORD *a2)
{
  unsigned __int64 *v2; // rax
  unsigned __int64 v3; // rbx
  _QWORD *v4; // r14
  _QWORD *v5; // r15
  unsigned int v6; // eax
  __int64 v7; // r9
  _QWORD *v8; // rax
  unsigned __int8 v9; // cl
  _QWORD *v10; // rax
  __int64 v12; // [rsp+0h] [rbp-70h]
  __int64 v13; // [rsp+8h] [rbp-68h]
  unsigned __int64 v14; // [rsp+10h] [rbp-60h]
  _QWORD *v15; // [rsp+18h] [rbp-58h]
  _BYTE v16[80]; // [rsp+20h] [rbp-50h] BYREF

  v2 = (unsigned __int64 *)a2[10];
  v3 = a2[9];
  v13 = a2[6];
  v14 = v2[1];
  if ( *v2 )
    sub_1296350(a1, *v2);
  v4 = (_QWORD *)sub_12A4D50(a1, "for.cond", 0, 0);
  v15 = (_QWORD *)sub_12A4D50(a1, "for.end", 0, 0);
  v5 = (_QWORD *)sub_12A4D50(a1, "for.body", 0, 0);
  v12 = *(_QWORD *)(a1[4] + 384);
  if ( !dword_4D04658 )
  {
    sub_12A0360(v12, *(unsigned int *)a2);
    sub_12A0660(v12, a1 + 6);
  }
  sub_1290AF0(a1, v4, 0);
  if ( v13 )
    sub_127FEC0((__int64)a1, v13);
  else
    sub_159C4F0(a1[5]);
  v6 = sub_12905B0(v3, 0);
  sub_12A4DB0(a1, v7, v5, v15, v6);
  sub_1290AF0(a1, v5, 0);
  if ( v3 )
    sub_1296350(a1, v3);
  if ( v14 )
  {
    sub_1290930((__int64)a1, (unsigned int *)(v14 + 36));
    sub_127C770((_QWORD *)(v14 + 36));
    v8 = (_QWORD *)sub_12A4D50(a1, "for.inc", 0, 0);
    sub_1290AF0(a1, v8, 0);
    v9 = 0;
    if ( (*(_BYTE *)(*(_QWORD *)v14 + 140LL) & 0xFB) == 8 )
      v9 = (sub_8D4C10(*(_QWORD *)v14, dword_4F077C4 != 2) & 2) != 0;
    sub_1280010((__int64)v16, a1, (__int64 *)v14, v9);
  }
  v10 = sub_12909B0(a1, (__int64)v4);
  if ( v10 && a2[8] )
    sub_1291160((__int64)a1, (__int64)v10, (__int64)a2);
  if ( !dword_4D04658 )
    sub_129F180(v12, a1 + 6);
  return sub_1290AF0(a1, v15, 1);
}
