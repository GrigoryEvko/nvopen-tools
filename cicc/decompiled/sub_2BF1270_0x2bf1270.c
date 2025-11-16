// Function: sub_2BF1270
// Address: 0x2bf1270
//
__int64 __fastcall sub_2BF1270(__int64 a1, int *a2, __int64 a3)
{
  bool v3; // zf
  int *v4; // r13
  __int64 v5; // r12
  unsigned __int8 v6; // al
  char v7; // bl
  int v8; // r14d
  int v10; // [rsp+8h] [rbp-48h]
  char v11; // [rsp+Eh] [rbp-42h]
  unsigned __int8 v12; // [rsp+Fh] [rbp-41h]
  __int64 v13; // [rsp+10h] [rbp-40h] BYREF
  int v14; // [rsp+18h] [rbp-38h] BYREF
  char v15; // [rsp+1Ch] [rbp-34h]

  v3 = *(_QWORD *)(a1 + 16) == 0;
  v13 = *(_QWORD *)a2;
  if ( v3 )
LABEL_10:
    sub_4263D6(a1, a2, a3);
  v4 = a2;
  a2 = (int *)&v13;
  v5 = a1;
  v6 = (*(__int64 (__fastcall **)(__int64, __int64 *))(a1 + 24))(a1, &v13);
  v7 = *((_BYTE *)v4 + 4);
  v12 = v6;
  v8 = 2 * *v4;
  v10 = v4[2];
  v11 = *((_BYTE *)v4 + 12);
  while ( v8 != v10 || v7 != v11 )
  {
    v3 = *(_QWORD *)(v5 + 16) == 0;
    v14 = v8;
    v15 = v7;
    if ( v3 )
      goto LABEL_10;
    a2 = &v14;
    a1 = v5;
    if ( v12 != (*(unsigned __int8 (__fastcall **)(__int64, int *))(v5 + 24))(v5, &v14) )
    {
      v4[2] = v8;
      *((_BYTE *)v4 + 12) = v7;
      return v12;
    }
    v8 *= 2;
  }
  return v12;
}
