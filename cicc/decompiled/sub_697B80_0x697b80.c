// Function: sub_697B80
// Address: 0x697b80
//
__int64 __fastcall sub_697B80(__int64 a1, unsigned int a2, int a3, int a4, int a5, int a6, __int64 a7)
{
  __int64 v10; // rax
  __int64 v11; // r12
  char v12; // al
  __int64 v13; // rdx
  __int64 i; // rax
  __int64 v16; // r13
  char v18[4]; // [rsp+14h] [rbp-DCh] BYREF
  __int64 v19; // [rsp+18h] [rbp-D8h] BYREF
  char v20[208]; // [rsp+20h] [rbp-D0h] BYREF

  sub_6E1DD0(&v19);
  sub_6E1E00(4, v20, 0, 1);
  v10 = sub_841B50(a1, a2, a3, a4, a5, a6, (__int64)v18, 0, a7);
  v11 = v10;
  if ( !a3 || !v10 )
    goto LABEL_6;
  v12 = *(_BYTE *)(v10 + 80);
  v13 = v11;
  if ( v12 == 16 )
  {
    v13 = **(_QWORD **)(v11 + 88);
    v12 = *(_BYTE *)(v13 + 80);
    if ( v12 != 24 )
    {
LABEL_5:
      if ( v12 != 10 )
        goto LABEL_6;
      goto LABEL_7;
    }
  }
  else if ( v12 != 24 )
  {
    goto LABEL_5;
  }
  v13 = *(_QWORD *)(v13 + 88);
  if ( *(_BYTE *)(v13 + 80) != 10 )
    goto LABEL_6;
LABEL_7:
  for ( i = *(_QWORD *)(*(_QWORD *)(v13 + 88) + 152LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v16 = *(_QWORD *)(**(_QWORD **)(i + 168) + 8LL);
  a1 = v16;
  if ( (unsigned int)sub_8D2FB0(v16) )
  {
    a1 = v16;
    if ( !(unsigned int)sub_8D4D20(v16) )
      v11 = 0;
  }
LABEL_6:
  sub_6E2B30(a1, a2);
  sub_6E1DF0(v19);
  return v11;
}
