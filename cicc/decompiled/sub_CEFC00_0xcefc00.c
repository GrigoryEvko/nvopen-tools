// Function: sub_CEFC00
// Address: 0xcefc00
//
unsigned __int8 *__fastcall sub_CEFC00(unsigned __int8 *a1, unsigned __int8 **a2)
{
  char v2; // bl
  unsigned __int8 *v3; // r12
  unsigned __int8 v4; // al
  unsigned __int8 *v5; // rax
  char v6; // dl
  _QWORD *v7; // rbx
  _QWORD *v8; // rdi
  __int64 v10; // rax
  __int64 v11; // rax
  unsigned __int8 *v12; // [rsp+8h] [rbp-68h] BYREF
  void *s; // [rsp+10h] [rbp-60h] BYREF
  __int64 v14; // [rsp+18h] [rbp-58h]
  _QWORD *v15; // [rsp+20h] [rbp-50h]
  __int64 v16; // [rsp+28h] [rbp-48h]
  int v17; // [rsp+30h] [rbp-40h]
  __int64 v18; // [rsp+38h] [rbp-38h]
  _QWORD v19[6]; // [rsp+40h] [rbp-30h] BYREF

  v2 = (char)a2;
  s = v19;
  v14 = 1;
  v15 = 0;
  v16 = 0;
  v17 = 1065353216;
  v18 = 0;
  v19[0] = 0;
  v3 = sub_BD3990(a1, (__int64)a2);
  v12 = v3;
  v4 = *v3;
  if ( *v3 == 84 )
    goto LABEL_28;
  while ( 1 )
  {
    if ( v4 <= 0x1Cu )
    {
      if ( !v2 || v4 != 5 || *((_WORD *)v3 + 1) != 34 )
        goto LABEL_8;
      goto LABEL_6;
    }
    if ( v4 != 85 )
    {
      if ( !v2 || v4 != 63 )
        goto LABEL_8;
      goto LABEL_6;
    }
    v10 = *((_QWORD *)v3 - 4);
    if ( !v10 || *(_BYTE *)v10 || *(_QWORD *)(v10 + 24) != *((_QWORD *)v3 + 10) || (*(_BYTE *)(v10 + 33) & 0x20) == 0 )
      goto LABEL_8;
    if ( !sub_CEA1D0(*(_DWORD *)(v10 + 36)) )
    {
      v11 = *((_QWORD *)v3 - 4);
      if ( !v11 || *(_BYTE *)v11 || *(_QWORD *)(v11 + 24) != *((_QWORD *)v3 + 10) )
        BUG();
      if ( *(_DWORD *)(v11 + 36) != 8170 )
        break;
    }
LABEL_6:
    v5 = sub_BD3990(*(unsigned __int8 **)&v3[-32 * (*((_DWORD *)v3 + 1) & 0x7FFFFFF)], (__int64)a2);
    a2 = &v12;
    v12 = v5;
    sub_CEF0C0(&s, &v12, 1);
    if ( !v6 )
      break;
    v3 = v12;
    v4 = *v12;
    if ( *v12 == 84 )
    {
LABEL_28:
      a2 = &v12;
      sub_CEF0C0(&s, &v12, 1);
      v3 = v12;
      v4 = *v12;
    }
  }
  v3 = v12;
LABEL_8:
  v7 = v15;
  while ( v7 )
  {
    v8 = v7;
    v7 = (_QWORD *)*v7;
    j_j___libc_free_0(v8, 16);
  }
  memset(s, 0, 8 * v14);
  v16 = 0;
  v15 = 0;
  if ( s != v19 )
    j_j___libc_free_0(s, 8 * v14);
  return v3;
}
