// Function: sub_C3BE30
// Address: 0xc3be30
//
__int64 __fastcall sub_C3BE30(_BYTE *a1, __int64 *a2)
{
  char *v2; // rax
  char v3; // al
  unsigned int v4; // r14d
  char v5; // al
  char v6; // dl
  bool v7; // r15
  char v8; // r8
  __int64 v9; // rax
  char v10; // dl
  int v12; // r14d
  int v13; // r14d
  _BYTE v14[20]; // [rsp+10h] [rbp-90h] BYREF
  char v15; // [rsp+24h] [rbp-7Ch]
  _QWORD v16[4]; // [rsp+30h] [rbp-70h] BYREF
  _QWORD v17[10]; // [rsp+50h] [rbp-50h] BYREF

  v2 = (char *)sub_C94E20(qword_4F863F0);
  if ( v2 )
    v3 = *v2;
  else
    v3 = qword_4F863F0[2];
  if ( v3 && *(_DWORD **)a1 == dword_3F657C0 )
  {
    sub_C36070((__int64)a1, 0, 0, 0);
    return 1;
  }
  v4 = sub_C39530(a1, a2);
  v5 = a1[20];
  v6 = v5 & 7;
  v7 = (v5 & 8) != 0;
  if ( (v5 & 6) == 0 )
  {
LABEL_12:
    if ( v6 != 3 )
      return v4;
    goto LABEL_7;
  }
  if ( v6 != 3 )
  {
    while ( 1 )
    {
      v10 = *((_BYTE *)a2 + 20);
      if ( (v10 & 6) == 0 || (v10 & 7) == 3 )
      {
        v6 = v5 & 7;
        goto LABEL_12;
      }
      if ( !(unsigned int)sub_C37580((__int64)a1, (__int64)a2) )
        break;
      v12 = sub_C3BD20((__int64)a1);
      v13 = v12 - sub_C3BD20((__int64)a2);
      sub_C33EB0(v17, a2);
      sub_C3BDC0((__int64)v14, (__int64)v17, v13, 1);
      sub_C338F0((__int64)v17);
      if ( (v15 & 7) == 1 || !(unsigned int)sub_C37580((__int64)a1, (__int64)v14) )
      {
        sub_C33EB0(v16, a2);
        sub_C3BDC0((__int64)v17, (__int64)v16, v13 - 1, 1);
        sub_C33870((__int64)v14, (__int64)v17);
        sub_C338F0((__int64)v17);
        sub_C338F0((__int64)v16);
      }
      v15 = a1[20] & 8 | v15 & 0xF7;
      v4 = sub_C3B1F0((__int64)a1, (__int64)v14, 1);
      if ( !*(_BYTE *)(*(_QWORD *)a1 + 24LL) && sub_C34230((__int64)a1) )
      {
        sub_C338F0((__int64)v14);
        break;
      }
      sub_C338F0((__int64)v14);
      v5 = a1[20];
      v6 = v5 & 7;
      if ( (v5 & 6) == 0 )
        goto LABEL_12;
      if ( v6 == 3 )
        goto LABEL_7;
    }
    v6 = a1[20] & 7;
    goto LABEL_12;
  }
LABEL_7:
  v8 = a1[20] & 0xF7 | (8 * v7);
  v9 = *(_QWORD *)a1;
  a1[20] = v8;
  if ( *(_DWORD *)(v9 + 20) == 2 )
    a1[20] = v8 & 0xF7;
  return v4;
}
