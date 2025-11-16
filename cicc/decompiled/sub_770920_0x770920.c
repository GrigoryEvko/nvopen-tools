// Function: sub_770920
// Address: 0x770920
//
__int64 __fastcall sub_770920(__int64 a1, unsigned __int64 *a2, __int64 a3, unsigned int a4)
{
  __int64 v6; // rdi
  unsigned __int64 v7; // rax
  __int64 v8; // rcx
  int v9; // r13d
  unsigned __int64 v10; // rbx
  __int64 v11; // rax
  char i; // dl
  __int64 v13; // r13
  __int64 v14; // r14
  __int64 *v15; // r15
  unsigned int j; // edx
  __int64 v17; // rax
  __int64 v18; // rax
  char k; // dl
  _QWORD *v20; // rax
  __int64 v21; // rbx
  __int64 result; // rax
  _QWORD *v23; // rax
  _QWORD *v24; // rdx
  _QWORD *v25; // rax
  __int64 v26; // [rsp+8h] [rbp-128h] BYREF
  unsigned __int64 *v27; // [rsp+10h] [rbp-120h]
  char v28; // [rsp+18h] [rbp-118h] BYREF

  v6 = a4;
  v7 = *a2;
  v26 = 0;
  v27 = a2;
  v8 = *(_QWORD *)(v7 + 96);
  if ( v8 )
  {
    a2 = (unsigned __int64 *)&v28;
    v9 = 1;
    do
    {
      if ( *(_BYTE *)(v8 + 80) != 8 )
        goto LABEL_10;
      v10 = *(_QWORD *)(v8 + 88);
      if ( !(_DWORD)v6 )
      {
        v11 = *(_QWORD *)(v10 + 120);
        for ( i = *(_BYTE *)(v11 + 140); i == 12; i = *(_BYTE *)(v11 + 140) )
          v11 = *(_QWORD *)(v11 + 160);
        if ( i != 11 )
        {
LABEL_10:
          LODWORD(v13) = v9 - 1;
          v10 = (unsigned __int64)(&v27)[(int)v13];
          goto LABEL_11;
        }
      }
      ++v9;
      *a2 = v10;
      if ( v9 == 30 )
        return 0;
      v8 = *(_QWORD *)(v8 + 96);
      ++a2;
    }
    while ( v8 );
    LODWORD(v13) = v9 - 1;
  }
  else
  {
    v10 = (unsigned __int64)a2;
    LODWORD(v13) = 0;
  }
LABEL_11:
  v13 = (int)v13;
  v14 = 0;
  v15 = &v26;
  if ( *(_BYTE *)(a3 + 140) == 11 )
    goto LABEL_21;
  while ( (_DWORD)v13 )
  {
    while ( 1 )
    {
      a2 = (unsigned __int64 *)qword_4F08380;
      for ( j = qword_4F08388 & (v10 >> 3); ; j = qword_4F08388 & (j + 1) )
      {
        v17 = qword_4F08380 + 16LL * j;
        v6 = *(_QWORD *)v17;
        if ( *(_QWORD *)v17 == v10 )
          break;
        if ( !v6 )
          goto LABEL_18;
      }
      v14 = (unsigned int)(*(_DWORD *)(v17 + 8) + v14);
LABEL_18:
      v18 = *(_QWORD *)(v10 + 120);
      for ( k = *(_BYTE *)(v18 + 140); k == 12; k = *(_BYTE *)(v18 + 140) )
        v18 = *(_QWORD *)(v18 + 160);
      v10 = *(&v26 + v13--);
      if ( k != 11 )
        break;
LABEL_21:
      v20 = (_QWORD *)sub_7708D0(v6, a2);
      *v15 = (__int64)v20;
      *v20 = 0;
      *(_QWORD *)(*v15 + 16) = v10;
      *(_QWORD *)(*v15 + 24) = *(_QWORD *)a1 + (unsigned int)v14;
      v15 = (__int64 *)*v15;
      if ( !(_DWORD)v13 )
        goto LABEL_22;
    }
  }
LABEL_22:
  v21 = v26;
  *(_QWORD *)a1 += v14;
  result = 1;
  if ( v21 )
  {
    if ( (*(_BYTE *)(a1 + 8) & 4) != 0 )
    {
      v23 = **(_QWORD ***)(a1 + 16);
      do
      {
        v24 = v23;
        v23 = (_QWORD *)*v23;
      }
      while ( v23 );
    }
    else
    {
      v25 = (_QWORD *)sub_7708D0(v6, a2);
      *(_QWORD *)(a1 + 16) = v25;
      v24 = v25;
      *v25 = 0;
      v25[2] = 0;
      v25[3] = 0;
      *(_BYTE *)(a1 + 8) |= 4u;
    }
    *v24 = v21;
    return 1;
  }
  return result;
}
