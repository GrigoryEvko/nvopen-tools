// Function: sub_2252D40
// Address: 0x2252d40
//
char *__fastcall sub_2252D40(__int64 a1, char *a2, __int64 a3)
{
  __int64 v3; // rax
  char v6; // di
  char *v7; // r13
  char v8; // al
  char *v9; // rdx
  __int64 v10; // rdi
  int v11; // ecx
  char v12; // si
  unsigned __int64 v13; // rax
  char *v14; // r8
  __int64 v15; // rsi
  int v16; // ecx
  char v17; // dl
  unsigned __int64 v18; // rax
  char *v20; // rax
  char *v21; // r13
  char v22; // al

  v3 = 0;
  if ( a1 )
    v3 = sub_39F7FD0(a1);
  *(_QWORD *)a3 = v3;
  v6 = *a2;
  v7 = a2 + 1;
  if ( *a2 == -1 )
  {
    *(_QWORD *)(a3 + 8) = v3;
    v8 = *v7;
    v9 = a2 + 2;
    *(_BYTE *)(a3 + 40) = *v7;
    if ( v8 != -1 )
      goto LABEL_5;
LABEL_12:
    *(_QWORD *)(a3 + 24) = 0;
    goto LABEL_8;
  }
  v20 = (char *)sub_2252CC0(v6, a1);
  v21 = sub_2252A40(v6, v20, v7, (unsigned __int64 *)(a3 + 8));
  v22 = *v21;
  v9 = v21 + 1;
  *(_BYTE *)(a3 + 40) = *v21;
  if ( v22 == -1 )
    goto LABEL_12;
LABEL_5:
  v10 = 0;
  v11 = 0;
  do
  {
    v12 = *v9++;
    v13 = (unsigned __int64)(v12 & 0x7F) << v11;
    v11 += 7;
    v10 |= v13;
  }
  while ( v12 < 0 );
  *(_QWORD *)(a3 + 24) = &v9[v10];
LABEL_8:
  v14 = v9 + 1;
  v15 = 0;
  v16 = 0;
  *(_BYTE *)(a3 + 41) = *v9;
  do
  {
    v17 = *v14++;
    v18 = (unsigned __int64)(v17 & 0x7F) << v16;
    v16 += 7;
    v15 |= v18;
  }
  while ( v17 < 0 );
  *(_QWORD *)(a3 + 32) = &v14[v15];
  return v14;
}
