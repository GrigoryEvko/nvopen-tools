// Function: sub_FDD8D0
// Address: 0xfdd8d0
//
__int64 __fastcall sub_FDD8D0(__int64 a1, __int64 a2, __int64 *a3)
{
  unsigned int v6; // ebx
  const char *v7; // rax
  size_t v8; // rdx
  char *v9; // rdi
  unsigned __int8 *v10; // rsi
  unsigned __int64 v11; // rax
  _QWORD *v12; // r8
  bool v13; // cc
  __int64 v14; // rdx
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  _WORD *v18; // rdx
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rax
  __int64 v21; // rax
  size_t v22; // [rsp+8h] [rbp-98h]
  unsigned __int64 v23[2]; // [rsp+10h] [rbp-90h] BYREF
  void (__fastcall *v24)(unsigned __int64 *, unsigned __int64 *, __int64); // [rsp+20h] [rbp-80h]
  void (__fastcall *v25)(unsigned __int64 *, _QWORD *); // [rsp+28h] [rbp-78h]
  _QWORD v26[3]; // [rsp+30h] [rbp-70h] BYREF
  __int64 v27; // [rsp+48h] [rbp-58h]
  void *dest; // [rsp+50h] [rbp-50h]
  __int64 v29; // [rsp+58h] [rbp-48h]
  __int64 v30; // [rsp+60h] [rbp-40h]

  if ( unk_4F8DC28 == 1 )
    v6 = 3;
  else
    v6 = dword_4F8E068;
  *(_BYTE *)(a1 + 16) = 0;
  *(_QWORD *)a1 = a1 + 16;
  v29 = 0x100000000LL;
  *(_QWORD *)(a1 + 8) = 0;
  v26[1] = 0;
  v26[0] = &unk_49DD210;
  v26[2] = 0;
  v27 = 0;
  dest = 0;
  v30 = a1;
  sub_CB5980((__int64)v26, 0, 0, 0);
  v7 = sub_BD5D20(a2);
  v9 = (char *)dest;
  v10 = (unsigned __int8 *)v7;
  v11 = v27 - (_QWORD)dest;
  if ( v27 - (__int64)dest < v8 )
  {
    v21 = sub_CB6200((__int64)v26, v10, v8);
    v9 = *(char **)(v21 + 32);
    v12 = (_QWORD *)v21;
    v11 = *(_QWORD *)(v21 + 24) - (_QWORD)v9;
LABEL_5:
    if ( v11 > 2 )
      goto LABEL_6;
    goto LABEL_18;
  }
  v12 = v26;
  if ( !v8 )
    goto LABEL_5;
  v22 = v8;
  memcpy(dest, v10, v8);
  v12 = v26;
  v9 = (char *)dest + v22;
  v19 = v27 - ((_QWORD)dest + v22);
  dest = (char *)dest + v22;
  if ( v19 > 2 )
  {
LABEL_6:
    v9[2] = 32;
    *(_WORD *)v9 = 14880;
    v12[4] += 3LL;
    v13 = v6 <= 2;
    if ( v6 != 2 )
      goto LABEL_7;
LABEL_19:
    v20 = sub_FDD860(a3, a2);
    sub_CB59D0((__int64)v26, v20);
    goto LABEL_12;
  }
LABEL_18:
  sub_CB6200((__int64)v12, (unsigned __int8 *)" : ", 3u);
  v13 = v6 <= 2;
  if ( v6 == 2 )
    goto LABEL_19;
LABEL_7:
  if ( v13 )
  {
    if ( !v6 )
      BUG();
    sub_FDD8A0(v23, a3, a2);
    if ( !v24 )
      sub_4263D6(v23, a3, v14);
    v25(v23, v26);
    if ( v24 )
      v24(v23, v23, 3);
  }
  else if ( v6 == 3 )
  {
    v16 = sub_FDD2C0(a3, a2, 0);
    v23[1] = v17;
    v23[0] = v16;
    if ( (_BYTE)v17 )
    {
      sub_CB59D0((__int64)v26, v23[0]);
    }
    else
    {
      v18 = dest;
      if ( (unsigned __int64)(v27 - (_QWORD)dest) <= 6 )
      {
        sub_CB6200((__int64)v26, "Unknown", 7u);
      }
      else
      {
        *(_DWORD *)dest = 1852534357;
        v18[2] = 30575;
        *((_BYTE *)v18 + 6) = 110;
        dest = (char *)dest + 7;
      }
    }
  }
LABEL_12:
  v26[0] = &unk_49DD210;
  sub_CB5840((__int64)v26);
  return a1;
}
