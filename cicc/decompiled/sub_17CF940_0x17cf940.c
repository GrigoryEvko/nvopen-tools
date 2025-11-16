// Function: sub_17CF940
// Address: 0x17cf940
//
__int64 __fastcall sub_17CF940(_QWORD *a1, __int64 *a2, _BYTE *a3, __int64 a4, char a5)
{
  __int64 v9; // rbx
  __int64 v10; // rdi
  unsigned __int64 v11; // rsi
  unsigned int v12; // eax
  unsigned int v13; // r9d
  __int64 *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // r14
  __int64 v21; // rax
  __int64 **v22; // rax
  __int64 v23; // rax
  int v24; // eax
  __int64 v25; // rdx
  char v26; // [rsp+8h] [rbp-68h]
  char v27; // [rsp+17h] [rbp-59h]
  unsigned int v28; // [rsp+18h] [rbp-58h]
  __int64 v30[2]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v31; // [rsp+30h] [rbp-40h]

  v9 = *(_QWORD *)a3;
  v10 = *(_QWORD *)a3;
  v27 = *(_BYTE *)(*(_QWORD *)a3 + 8LL);
  if ( v27 == 16 )
    v11 = *(_DWORD *)(v9 + 32) * (unsigned int)sub_16431D0(v10);
  else
    v11 = (unsigned int)sub_1643030(v10);
  if ( *(_BYTE *)(a4 + 8) != 16 )
  {
    v26 = *(_BYTE *)(a4 + 8);
    v12 = sub_1643030(a4);
    v13 = v12;
    if ( v12 == 1 && v11 > 1 )
      goto LABEL_6;
    if ( v26 == 11 )
    {
      if ( v27 != 11 )
      {
LABEL_11:
        v18 = a1[1];
        v28 = v13;
        v31 = 257;
        v19 = sub_1644C60(*(_QWORD **)(v18 + 168), v11);
        v20 = sub_12AA3B0(a2, 0x2Fu, (__int64)a3, v19, (__int64)v30);
        v21 = a1[1];
        v31 = 257;
        v22 = (__int64 **)sub_1644C60(*(_QWORD **)(v21 + 168), v28);
        v23 = sub_17CE200(a2, v20, v22, a5, v30);
        v31 = 257;
        return sub_12AA3B0(a2, 0x2Fu, v23, a4, (__int64)v30);
      }
LABEL_18:
      v31 = 257;
      return sub_17CE200(a2, (__int64)a3, (__int64 **)a4, a5, v30);
    }
    goto LABEL_15;
  }
  v24 = sub_16431D0(a4);
  v25 = *(_QWORD *)(a4 + 32);
  v12 = v25 * v24;
  v13 = v12;
  if ( v12 != 1 || v11 <= 1 )
  {
    if ( v27 == 16 )
    {
      if ( *(_DWORD *)(v9 + 32) != (_DWORD)v25 )
        goto LABEL_11;
      goto LABEL_18;
    }
LABEL_15:
    v13 = v12;
    goto LABEL_11;
  }
LABEL_6:
  v31 = 257;
  v14 = sub_17CD8D0(a1, v9);
  v16 = (__int64)v14;
  if ( v14 )
    v16 = sub_15A06D0((__int64 **)v14, v9, v15, (__int64)v14);
  return sub_12AA0C0(a2, 0x21u, a3, v16, (__int64)v30);
}
