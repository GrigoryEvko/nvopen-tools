// Function: sub_70DD40
// Address: 0x70dd40
//
__int64 __fastcall sub_70DD40(__int64 a1, __int64 **a2, int a3)
{
  char v5; // dl
  __int64 v6; // r12
  char v7; // al
  __int64 result; // rax
  __int64 *v9; // rax
  __int64 *v10; // r13
  __int64 *v11; // rbx
  __int64 **v12; // r15
  char v13; // si
  bool v14; // dl
  __int64 v15; // r15
  __int64 v16; // r10
  __int64 v17; // r9
  unsigned int v18; // r8d
  __int64 i; // rbx
  int v20; // eax
  __int64 *v21; // rdi
  int v22; // edx
  __int64 v23; // rdi
  __int64 v24; // [rsp+8h] [rbp-58h]
  __int64 v25; // [rsp+10h] [rbp-50h]
  unsigned int v26; // [rsp+1Ch] [rbp-44h]
  __int64 v27; // [rsp+28h] [rbp-38h] BYREF

  v5 = *(_BYTE *)(a1 + 140);
  if ( v5 == 12 )
  {
LABEL_2:
    v6 = a1;
    do
    {
      v6 = *(_QWORD *)(v6 + 160);
      v7 = *(_BYTE *)(v6 + 140);
    }
    while ( v7 == 12 );
  }
  else
  {
LABEL_16:
    v7 = v5;
    v6 = a1;
  }
  if ( (unsigned __int8)(v7 - 9) > 2u )
  {
    if ( !dword_4F077BC || (v5 & 0xFB) != 8 || (sub_8D4C10(a1, dword_4F077C4 != 2) & 2) == 0 )
    {
      switch ( *(_BYTE *)(v6 + 140) )
      {
        case 0:
        case 2:
        case 0xD:
        case 0xE:
          return 1;
        case 6:
          return (*(_BYTE *)(v6 + 168) & 1) == 0;
        case 8:
          a1 = sub_8D40F0(v6);
          goto LABEL_15;
        case 0xF:
          if ( (_DWORD)qword_4F077B4 )
            return 0;
          a1 = *(_QWORD *)(v6 + 160);
LABEL_15:
          v5 = *(_BYTE *)(a1 + 140);
          a3 = 1;
          a2 = 0;
          if ( v5 != 12 )
            goto LABEL_16;
          goto LABEL_2;
        default:
          return 0;
      }
    }
    return 0;
  }
  if ( !(_DWORD)qword_4F077B4 && dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(v6) )
    sub_8AE000(v6);
  if ( a3 && !(unsigned int)sub_8E3AD0(v6) )
    return 0;
  v27 = 0;
  v9 = 0;
  if ( unk_4F068A8 )
    v9 = &v27;
  v10 = 0;
  v11 = v9;
  v12 = **(__int64 ****)(v6 + 168);
  if ( v12 )
  {
    while ( 1 )
    {
      if ( ((_BYTE)v12[12] & 1) == 0 )
        goto LABEL_31;
      if ( v12[13] != v10 )
      {
        v13 = *(_BYTE *)(v6 + 140);
        result = 0;
        goto LABEL_28;
      }
      v21 = v12[5];
      if ( v21[20] )
      {
        v22 = sub_70DD40(v21, v11, 1);
        result = v22 != 0;
        if ( unk_4F068A8 )
          v23 = v27;
        else
          v23 = v12[5][16];
        v10 = (__int64 *)((char *)v10 + v23);
        v14 = v22 == 0;
      }
      else
      {
LABEL_31:
        v14 = 0;
        result = 1;
      }
      v12 = (__int64 **)*v12;
      if ( !v12 || v14 )
        goto LABEL_34;
    }
  }
  v14 = 0;
  v10 = 0;
  result = 1;
LABEL_34:
  v15 = *(_QWORD *)(v6 + 160);
  v13 = *(_BYTE *)(v6 + 140);
  if ( v15 && !v14 )
  {
    v16 = 0;
    v17 = 0;
    v18 = 0;
    while ( 1 )
    {
      for ( i = *(_QWORD *)(v15 + 120); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      if ( v13 == 11 )
      {
        if ( v17 && *(_QWORD *)(i + 128) != v16 )
        {
          result = 0;
          goto LABEL_52;
        }
        if ( (*(_BYTE *)(v15 + 144) & 4) != 0 )
        {
          result = (unsigned int)qword_4F077B4;
          if ( !(_DWORD)qword_4F077B4
            && *(unsigned __int8 *)(v15 + 137) < *(_QWORD *)(v6 + 128) * (unsigned __int64)dword_4F06BA0 )
          {
            goto LABEL_52;
          }
        }
      }
      if ( *(__int64 **)(v15 + 128) != v10 || *(unsigned __int8 *)(v15 + 136) != v18 )
      {
        result = 0;
        break;
      }
      v24 = v16;
      v25 = v17;
      v26 = v18;
      v20 = sub_70DD40(i, 0, 1);
      v18 = v26;
      v17 = v25;
      v13 = *(_BYTE *)(v6 + 140);
      v16 = v24;
      if ( v13 == 11 )
      {
        if ( (*(_BYTE *)(v15 + 144) & 4) != 0 )
          v16 = *(_QWORD *)(v6 + 128);
        else
          v16 = *(_QWORD *)(i + 128);
        v17 = v15;
      }
      else if ( (*(_BYTE *)(v15 + 144) & 4) != 0 )
      {
        v10 = (__int64 *)((char *)v10 + (v26 + *(unsigned __int8 *)(v15 + 137)) / dword_4F06BA0);
        v18 = (v26 + *(unsigned __int8 *)(v15 + 137)) % dword_4F06BA0;
      }
      else
      {
        v10 = (__int64 *)((char *)v10 + *(_QWORD *)(i + 128));
      }
      v15 = *(_QWORD *)(v15 + 112);
      if ( !v15 || !v20 )
      {
        result = v20 != 0;
        break;
      }
    }
  }
LABEL_28:
  if ( v13 == 11 )
  {
LABEL_52:
    if ( !*(_QWORD *)(v6 + 160) )
      return 0;
  }
  else if ( a2 )
  {
    *a2 = v10;
  }
  else if ( *(__int64 **)(v6 + 128) != v10 )
  {
    return 0;
  }
  return result;
}
