// Function: sub_67CC10
// Address: 0x67cc10
//
void __fastcall sub_67CC10(__int64 a1, __int64 a2, int *a3)
{
  int *v3; // r14
  _QWORD *v5; // r12
  __int64 v6; // rdx
  unsigned __int8 v7; // al
  __int64 v8; // rbx
  _QWORD *v9; // rax
  __int64 **v10; // r12
  char *v11; // rax
  const char *v12; // r15
  size_t v13; // rax
  __int64 v14; // rax
  __int64 v15; // rsi
  __int64 v16; // rdi
  int v17; // eax
  __int64 *v18; // rcx
  __int64 v19; // [rsp+0h] [rbp-50h]
  char v20; // [rsp+Fh] [rbp-41h]
  int v21; // [rsp+1Ch] [rbp-34h] BYREF

  v3 = a3;
  v5 = (_QWORD *)a2;
  if ( !a3 )
  {
    v21 = 0;
    v3 = &v21;
  }
  if ( a2 )
  {
    switch ( *(_BYTE *)(a2 + 80) )
    {
      case 4:
      case 5:
        v6 = *(_QWORD *)(*(_QWORD *)(a2 + 96) + 80LL);
        break;
      case 6:
        v6 = *(_QWORD *)(*(_QWORD *)(a2 + 96) + 32LL);
        break;
      case 9:
      case 0xA:
        v6 = *(_QWORD *)(*(_QWORD *)(a2 + 96) + 56LL);
        break;
      case 0x13:
      case 0x14:
      case 0x15:
      case 0x16:
        v6 = *(_QWORD *)(a2 + 88);
        break;
      default:
        v6 = 0;
        break;
    }
    v7 = *(_BYTE *)(a1 + 80);
    if ( v7 != 9 )
    {
      if ( v7 > 9u )
      {
        if ( (unsigned __int8)(v7 - 10) <= 1u )
        {
          v14 = *(_QWORD *)(a1 + 88);
          if ( (*(_BYTE *)(v14 + 195) & 8) != 0 )
          {
            if ( (*(_BYTE *)(a1 + 81) & 0x10) == 0 )
              goto LABEL_11;
            v8 = 0;
            v15 = **(_QWORD **)(a2 + 64);
            v9 = 0;
            if ( !v15 )
              goto LABEL_11;
            goto LABEL_30;
          }
          v8 = *(_QWORD *)(v14 + 240);
          v9 = *(_QWORD **)(v6 + 328);
          if ( (*(_BYTE *)(a1 + 81) & 0x10) == 0 )
            goto LABEL_17;
LABEL_29:
          v15 = **(_QWORD **)(a2 + 64);
          if ( !v15 )
            goto LABEL_17;
LABEL_30:
          v16 = **(_QWORD **)(a1 + 64);
          if ( (unsigned __int8)(*(_BYTE *)(v15 + 80) - 4) > 1u )
            goto LABEL_17;
          v5 = v9;
          if ( *(char *)(*(_QWORD *)(v15 + 88) + 177LL) >= 0 )
            goto LABEL_17;
          goto LABEL_47;
        }
        goto LABEL_58;
      }
      if ( v7 <= 5u )
      {
        if ( v7 > 3u )
        {
          v19 = v6;
          v20 = *(_BYTE *)(*(_QWORD *)(a1 + 88) + 178LL) & 1;
          v8 = sub_892330();
          if ( v20 )
            goto LABEL_11;
          v9 = *(_QWORD **)(v19 + 32);
LABEL_16:
          if ( (*(_BYTE *)(a1 + 81) & 0x10) == 0 )
            goto LABEL_17;
          goto LABEL_29;
        }
LABEL_58:
        sub_721090(a1);
      }
      if ( v7 != 7 )
        goto LABEL_58;
    }
    v9 = *(_QWORD **)(v6 + 232);
    v8 = **(_QWORD **)(*(_QWORD *)(a1 + 88) + 216LL);
    goto LABEL_16;
  }
  if ( (*(_BYTE *)(a1 + 81) & 0x10) == 0 )
    goto LABEL_11;
  v15 = sub_67C320(*(_QWORD *)(a1 + 64));
  if ( !v15 )
    goto LABEL_11;
  v16 = *v18;
  if ( (unsigned __int8)(*(_BYTE *)(v15 + 80) - 4) > 1u || *(char *)(*(_QWORD *)(v15 + 88) + 177LL) >= 0 )
    goto LABEL_11;
  v8 = 0;
LABEL_47:
  if ( (unsigned __int8)(*(_BYTE *)(v16 + 80) - 4) > 1u || (v9 = v5, *(char *)(*(_QWORD *)(v16 + 88) + 177LL) >= 0) )
  {
    sub_67CC10(v16, v15, v3);
    v9 = v5;
  }
LABEL_17:
  if ( v8 )
  {
    v10 = (__int64 **)*v9;
    if ( *v9 )
    {
      do
      {
        if ( *v3 )
        {
          sub_8238B0(qword_4D039E8, ", ", 2);
        }
        else
        {
          sub_8238B0(qword_4D039E8, " [", 2);
          v11 = sub_67C860(1460);
          sub_823910(qword_4D039E8, v11);
          *v3 = 1;
        }
        v12 = *(const char **)(*v10[1] + 8);
        v13 = strlen(v12);
        sub_8238B0(qword_4D039E8, v12, v13);
        sub_8238B0(qword_4D039E8, "=", 1);
        if ( *(_BYTE *)(v8 + 8) == 3 )
        {
          sub_8238B0(qword_4D039E8, "<", 1);
          v8 = *(_QWORD *)v8;
          v17 = 1;
          if ( v8 )
          {
            while ( *(_BYTE *)(v8 + 8) != 3 && (*(_BYTE *)(v8 + 24) & 8) != 0 )
            {
              if ( !v17 )
                sub_8238B0(qword_4D039E8, ", ", 2);
              sub_747370(v8, &qword_4CFFDC0);
              v8 = *(_QWORD *)v8;
              v17 = 0;
              if ( !v8 )
                goto LABEL_40;
            }
          }
          else
          {
LABEL_40:
            v8 = 0;
          }
          sub_8238B0(qword_4D039E8, ">", 1);
        }
        else
        {
          sub_747370(v8, &qword_4CFFDC0);
          v8 = *(_QWORD *)v8;
        }
        v10 = (__int64 **)*v10;
      }
      while ( v10 );
    }
  }
LABEL_11:
  if ( !a3 )
  {
    if ( *v3 )
      sub_8238B0(qword_4D039E8, "]", 1);
  }
}
