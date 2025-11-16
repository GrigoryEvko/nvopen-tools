// Function: sub_86B560
// Address: 0x86b560
//
_BYTE *__fastcall sub_86B560(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // r13
  _BYTE *v6; // rdx
  _BYTE *v7; // r8
  __int64 i; // r12
  char v9; // al
  char v10; // al
  _QWORD *v11; // r15
  _BYTE *v12; // rax
  _BYTE *v14; // [rsp+8h] [rbp-48h]
  _BYTE *v15; // [rsp+10h] [rbp-40h]
  __int64 v16; // [rsp+18h] [rbp-38h]

  v2 = 0;
  v3 = a1;
  v4 = *(_QWORD *)(a1 + 16);
  v5 = qword_4F5FD70;
  if ( a2 )
  {
    v2 = *(_QWORD *)(a2 + 16);
    v5 = a2;
  }
  if ( !v4 )
    return 0;
  v6 = 0;
  v7 = 0;
  while ( (*(_BYTE *)(v4 + 72) & 2) != 0 )
  {
LABEL_9:
    if ( v2 == v4 )
      v4 = v5;
    for ( i = v4; v3 != i; v6 = v12 )
    {
      while ( 1 )
      {
        v9 = *(_BYTE *)(v3 + 32);
        if ( v9 != 1 )
        {
          if ( v9 == 5 )
            v3 = *(_QWORD *)(v3 + 40);
          goto LABEL_15;
        }
        v10 = *(_BYTE *)(v3 + 56);
        if ( (v10 & 1) != 0 && (v10 & 2) == 0 )
          break;
LABEL_15:
        v3 = *(_QWORD *)(v3 + 8);
        if ( v3 == i )
          goto LABEL_21;
      }
      v15 = v7;
      v14 = v6;
      v16 = *(_QWORD *)(v3 + 48);
      v11 = sub_726700(21);
      *v11 = sub_72CBE0();
      v11[7] = v16;
      v12 = sub_732B10((__int64)v11);
      v7 = v15;
      if ( !v15 )
      {
        v6 = v12;
        v7 = v12;
        goto LABEL_15;
      }
      *((_QWORD *)v14 + 2) = v12;
      v3 = *(_QWORD *)(v3 + 8);
    }
LABEL_21:
    if ( v5 == i )
      return v7;
    v4 = *(_QWORD *)(i + 16);
    v3 = i;
  }
  while ( v2 != v4 && v5 != v4 )
  {
    v3 = v4;
    v4 = *(_QWORD *)(v4 + 16);
    if ( (*(_BYTE *)(v4 + 72) & 2) != 0 )
      goto LABEL_9;
  }
  return v7;
}
