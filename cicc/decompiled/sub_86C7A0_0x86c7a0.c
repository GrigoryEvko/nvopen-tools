// Function: sub_86C7A0
// Address: 0x86c7a0
//
__int64 __fastcall sub_86C7A0(__int64 a1, __int64 a2, unsigned __int8 *a3)
{
  __int64 v4; // r14
  __int64 *v6; // r12
  __int64 v7; // rbx
  __int64 result; // rax
  bool v9; // cc
  __int64 v10; // rcx
  __int64 v11; // r8
  unsigned __int8 v12; // r14
  unsigned __int8 v13; // al
  unsigned __int8 v14; // r9
  __int64 i; // rdx
  _DWORD *v16; // rax
  _QWORD *v17; // rdi
  __int64 *v18; // rax
  int v19; // ecx
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // rdx
  _QWORD *v23; // rax
  __int64 j; // rax
  __int64 v25; // r14
  int v26; // eax
  char v27; // al
  __int64 v28; // rax
  __int64 v29; // rax
  unsigned __int8 v30; // [rsp+Ch] [rbp-44h]
  __int64 v31; // [rsp+10h] [rbp-40h]
  __int64 v32; // [rsp+10h] [rbp-40h]
  __int64 v33; // [rsp+10h] [rbp-40h]
  __int64 v34; // [rsp+10h] [rbp-40h]
  __int64 v35; // [rsp+18h] [rbp-38h]
  __int64 v36; // [rsp+18h] [rbp-38h]
  __int64 v37; // [rsp+18h] [rbp-38h]
  __int64 v38; // [rsp+18h] [rbp-38h]

  v4 = 0;
  v6 = (__int64 *)a2;
  v7 = *(_QWORD *)a1;
  result = *(unsigned __int8 *)(*(_QWORD *)a1 + 32LL);
  v9 = (unsigned __int8)result <= 4u;
  if ( (_BYTE)result == 4 )
    goto LABEL_28;
LABEL_2:
  if ( !v9 )
  {
    if ( (_BYTE)result != 5 )
      goto LABEL_26;
    return result;
  }
  if ( (_BYTE)result )
  {
    if ( (_BYTE)result != 1 )
      goto LABEL_26;
    v10 = *(_QWORD *)(v7 + 48);
    v11 = *(_QWORD *)(v7 + 40);
    if ( !v10 || (*(_BYTE *)(v7 + 56) & 1) != 0 )
    {
      if ( *(_BYTE *)(v11 + 40) == 22 && !*(_BYTE *)(v11 + 72) )
        v10 = *(_QWORD *)(v11 + 80);
      v13 = *a3;
      if ( *a3 == 8 )
        goto LABEL_20;
      v14 = 8;
      v12 = 8;
LABEL_11:
      if ( v13 != 3 )
      {
        v30 = v14;
        v31 = v10;
        v35 = v11;
        sub_685910(*v6, (FILE *)a2);
        v14 = v30;
        v10 = v31;
        v11 = v35;
      }
      for ( i = *(_QWORD *)(v7 + 16); (*(_BYTE *)(i + 72) & 8) != 0; i = *(_QWORD *)(i + 16) )
        ;
      v32 = v10;
      v36 = v11;
      v16 = sub_67D910(v14, 0x222u, (_DWORD *)(i + 24));
      v10 = v32;
      v11 = v36;
      *v6 = (__int64)v16;
      *a3 = v12;
LABEL_20:
      if ( v10 )
      {
LABEL_21:
        v17 = (_QWORD *)*v6;
        if ( (*(_BYTE *)(v10 + 172) & 2) != 0 )
        {
          a2 = 2439;
          sub_67DDB0(v17, 2439, (_QWORD *)(v10 + 64));
        }
        else
        {
          a2 = 547;
          if ( v11 && *(_BYTE *)(v11 + 40) == 22 )
            a2 = 1033;
          sub_67E1D0(v17, a2, *(_QWORD *)v10);
        }
        goto LABEL_26;
      }
      a2 = 895;
      sub_67DDB0((_QWORD *)*v6, 895, (_QWORD *)v11);
LABEL_26:
      v4 = *(_QWORD *)v7;
      goto LABEL_27;
    }
    if ( *(_BYTE *)(v10 + 136) <= 2u )
      goto LABEL_26;
    if ( dword_4F077C4 != 2 )
      goto LABEL_9;
    if ( (*(_BYTE *)(*(_QWORD *)(qword_4F04C50 + 32LL) + 193LL) & 2) == 0 )
    {
      if ( HIDWORD(qword_4D0495C) )
      {
LABEL_9:
        v12 = 5;
        goto LABEL_10;
      }
      v25 = *(_QWORD *)(v10 + 120);
      v33 = *(_QWORD *)(v7 + 40);
      v37 = *(_QWORD *)(v7 + 48);
      v26 = sub_8D3410(v25);
      v10 = v37;
      v11 = v33;
      if ( v26 )
      {
        v34 = v37;
        v38 = v11;
        v29 = sub_8D40F0(v25);
        v10 = v34;
        v11 = v38;
        v25 = v29;
      }
      while ( 1 )
      {
        v27 = *(_BYTE *)(v25 + 140);
        if ( v27 != 12 )
          break;
        v25 = *(_QWORD *)(v25 + 160);
      }
      if ( (unsigned __int8)(v27 - 9) > 2u
        || (v28 = *(_QWORD *)(*(_QWORD *)v25 + 96LL), !*(_QWORD *)(v28 + 24))
        || (*(_BYTE *)(v28 + 177) & 2) != 0 )
      {
        if ( dword_4D04964 )
        {
          v12 = byte_4F07472[0];
          if ( byte_4F07472[0] == 3 )
            goto LABEL_26;
          goto LABEL_10;
        }
        goto LABEL_9;
      }
    }
    v12 = 8;
LABEL_10:
    v13 = *a3;
    v14 = v12;
    if ( v12 == *a3 )
      goto LABEL_21;
    goto LABEL_11;
  }
  v4 = **(_QWORD **)(v7 + 40);
  if ( !*(_QWORD *)(v7 + 48) )
    goto LABEL_75;
  a2 = (__int64)v6;
  result = sub_86C7A0(v7, v6, a3);
  if ( (*(_BYTE *)(v7 + 72) & 1) == 0 && !*(_QWORD *)(v7 + 64) )
  {
    v22 = *(_QWORD *)(v7 + 40);
    v23 = *(_QWORD **)(v7 + 8);
    if ( v23 )
    {
      *v23 = *(_QWORD *)v22;
      v23 = *(_QWORD **)(v7 + 8);
    }
    else
    {
      qword_4F5FD70 = *(_QWORD *)v22;
    }
    if ( *(_QWORD *)v22 )
      *(_QWORD *)(*(_QWORD *)v22 + 8LL) = v23;
    else
      qword_4F5FD68 = (__int64)v23;
    result = qword_4F5FD60;
    *(_QWORD *)v22 = qword_4F5FD60;
    qword_4F5FD60 = v7;
  }
  if ( *(_QWORD *)(a1 + 48) )
  {
LABEL_75:
    do
    {
LABEL_27:
      v7 = v4;
      result = *(unsigned __int8 *)(v4 + 32);
      v9 = (unsigned __int8)result <= 4u;
      if ( (_BYTE)result != 4 )
        goto LABEL_2;
LABEL_28:
      if ( *(_QWORD *)(a1 + 48) == v7 )
      {
        *(_QWORD *)(a1 + 48) = 0;
        if ( (*(_BYTE *)(a1 + 72) & 4) == 0 )
        {
          for ( j = *(_QWORD *)(a1 + 16); *(_QWORD *)(j + 48) == v7; j = *(_QWORD *)(j + 16) )
          {
            *(_QWORD *)(j + 48) = 0;
            if ( (*(_BYTE *)(j + 72) & 4) != 0 )
              break;
          }
        }
        v18 = *(__int64 **)(v7 + 8);
        v20 = *(_QWORD *)v7;
        v19 = 1;
        if ( v18 )
        {
LABEL_30:
          *v18 = v20;
          v21 = *(_QWORD *)v7;
          v18 = *(__int64 **)(v7 + 8);
          if ( !*(_QWORD *)v7 )
            goto LABEL_47;
          goto LABEL_31;
        }
      }
      else
      {
        v4 = *(_QWORD *)v7;
        v18 = *(__int64 **)(v7 + 8);
        v19 = 0;
        v20 = *(_QWORD *)v7;
        if ( v18 )
          goto LABEL_30;
      }
      qword_4F5FD70 = v20;
      v21 = *(_QWORD *)v7;
      if ( !*(_QWORD *)v7 )
      {
LABEL_47:
        qword_4F5FD68 = (__int64)v18;
        goto LABEL_32;
      }
LABEL_31:
      *(_QWORD *)(v21 + 8) = v18;
LABEL_32:
      result = qword_4F5FD60;
      *(_QWORD *)v7 = qword_4F5FD60;
      qword_4F5FD60 = v7;
    }
    while ( !v19 );
  }
  return result;
}
