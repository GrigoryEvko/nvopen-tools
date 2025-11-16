// Function: sub_B19C20
// Address: 0xb19c20
//
__int64 __fastcall sub_B19C20(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // r15
  __int64 v4; // r13
  unsigned int v5; // r14d
  __int64 v7; // rbx
  __int64 v8; // rdx
  int v9; // ecx
  char v10; // al
  __int64 v11; // rdx
  int v12; // [rsp+Ch] [rbp-34h]

  v3 = a2[1];
  v4 = *a2;
  v5 = sub_B19720(a1, v3, a3);
  if ( !(_BYTE)v5 )
    return v5;
  if ( sub_AA54C0(v3) )
    return v5;
  v7 = *(_QWORD *)(v3 + 16);
  if ( !v7 )
    return v5;
  while ( 1 )
  {
    v8 = *(_QWORD *)(v7 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v8 - 30) <= 0xAu )
      break;
    v7 = *(_QWORD *)(v7 + 8);
    if ( !v7 )
      return v5;
  }
  v9 = 0;
LABEL_10:
  v11 = *(_QWORD *)(v8 + 40);
  if ( v4 != v11 )
  {
    v12 = v9;
    v10 = sub_B19720(a1, v3, v11);
    v9 = v12;
    if ( v10 )
      goto LABEL_8;
    return 0;
  }
  if ( v9 )
    return 0;
  v9 = 1;
LABEL_8:
  while ( 1 )
  {
    v7 = *(_QWORD *)(v7 + 8);
    if ( !v7 )
      return v5;
    v8 = *(_QWORD *)(v7 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v8 - 30) <= 0xAu )
      goto LABEL_10;
  }
}
