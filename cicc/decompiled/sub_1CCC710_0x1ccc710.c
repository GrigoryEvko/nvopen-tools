// Function: sub_1CCC710
// Address: 0x1ccc710
//
void __fastcall sub_1CCC710(__int64 **a1, void *a2, size_t a3, unsigned int a4)
{
  size_t v5; // rdx
  char *v8; // r15
  __int64 v9; // r15
  __int64 *v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rax
  _QWORD *v13; // rax
  _BYTE *v14; // rsi
  _BYTE *v15; // rsi
  __int64 v16; // rax
  __int64 v17; // [rsp-60h] [rbp-60h] BYREF
  __int64 *v18; // [rsp-58h] [rbp-58h] BYREF
  _BYTE *v19; // [rsp-50h] [rbp-50h]
  _BYTE *v20; // [rsp-48h] [rbp-48h]

  if ( a1 )
  {
    v5 = 0;
    v8 = off_4CD4980[0];
    if ( off_4CD4980[0] )
      v5 = strlen(off_4CD4980[0]);
    v9 = sub_1632440((__int64)a1, v8, v5);
    if ( v9 && !(unsigned __int8)sub_1CCACD0((__int64)a1, a2, a3, a4) )
    {
      v10 = *a1;
      v18 = 0;
      v19 = 0;
      v20 = 0;
      v17 = sub_161FF10(v10, a2, a3);
      sub_1273E00((__int64)&v18, 0, &v17);
      v11 = sub_1643350(*a1);
      v12 = sub_159C470(v11, a4, 0);
      if ( *(_BYTE *)(v12 + 16) == 19 )
        v13 = *(_QWORD **)(v12 + 24);
      else
        v13 = sub_1624210(v12);
      v17 = (__int64)v13;
      v14 = v19;
      if ( v19 == v20 )
      {
        sub_1273E00((__int64)&v18, v19, &v17);
        v15 = v19;
      }
      else
      {
        if ( v19 )
        {
          *(_QWORD *)v19 = v13;
          v14 = v19;
        }
        v15 = v14 + 8;
        v19 = v15;
      }
      v16 = sub_1627350(*a1, v18, (__int64 *)((v15 - (_BYTE *)v18) >> 3), 0, 1);
      sub_1623CA0(v9, v16);
      if ( v18 )
        j_j___libc_free_0(v18, v20 - (_BYTE *)v18);
    }
  }
}
