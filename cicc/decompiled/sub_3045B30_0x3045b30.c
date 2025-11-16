// Function: sub_3045B30
// Address: 0x3045b30
//
__int64 __fastcall sub_3045B30(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v7; // rax
  size_t *v8; // rsi
  __int64 v9; // rax
  _DWORD *v10; // rdi
  size_t v11; // r15
  unsigned __int8 *v12; // rsi
  unsigned __int64 v13; // rax
  _QWORD *v14; // rdi
  _QWORD v16[3]; // [rsp+0h] [rbp-70h] BYREF
  __int64 v17; // [rsp+18h] [rbp-58h]
  void *dest; // [rsp+20h] [rbp-50h]
  __int64 v19; // [rsp+28h] [rbp-48h]
  __int64 v20; // [rsp+30h] [rbp-40h]

  v19 = 0x100000000LL;
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  v16[0] = &unk_49DD210;
  v20 = a1;
  v16[1] = 0;
  v16[2] = 0;
  v17 = 0;
  dest = 0;
  sub_CB5980((__int64)v16, 0, 0, 0);
  v7 = sub_23CF390(*(_QWORD *)(a2 + 8), a3);
  if ( (*(_BYTE *)(v7 + 8) & 1) == 0 )
    goto LABEL_4;
  v8 = *(size_t **)(v7 - 8);
  v9 = v17;
  v10 = dest;
  v11 = *v8;
  v12 = (unsigned __int8 *)(v8 + 3);
  if ( v17 - (__int64)dest < v11 )
  {
    sub_CB6200((__int64)v16, v12, v11);
LABEL_4:
    v10 = dest;
    v9 = v17;
    goto LABEL_5;
  }
  if ( v11 )
  {
    memcpy(dest, v12, v11);
    v10 = (char *)dest + v11;
    dest = v10;
    v13 = v17 - (_QWORD)v10;
    if ( a4 >= 0 )
      goto LABEL_6;
LABEL_12:
    if ( v13 <= 6 )
    {
      sub_CB6200((__int64)v16, "_vararg", 7u);
    }
    else
    {
      *v10 = 1918989919;
      *((_WORD *)v10 + 2) = 29281;
      *((_BYTE *)v10 + 6) = 103;
      dest = (char *)dest + 7;
    }
    goto LABEL_9;
  }
LABEL_5:
  v13 = v9 - (_QWORD)v10;
  if ( a4 < 0 )
    goto LABEL_12;
LABEL_6:
  if ( v13 <= 6 )
  {
    v14 = (_QWORD *)sub_CB6200((__int64)v16, "_param_", 7u);
  }
  else
  {
    *v10 = 1918988383;
    *((_WORD *)v10 + 2) = 28001;
    *((_BYTE *)v10 + 6) = 95;
    v14 = v16;
    dest = (char *)dest + 7;
  }
  sub_CB59F0((__int64)v14, a4);
LABEL_9:
  v16[0] = &unk_49DD210;
  sub_CB5840((__int64)v16);
  return a1;
}
