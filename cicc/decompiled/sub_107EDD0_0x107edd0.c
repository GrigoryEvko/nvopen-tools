// Function: sub_107EDD0
// Address: 0x107edd0
//
__int64 __fastcall sub_107EDD0(__int64 a1, __int64 *a2)
{
  __int64 result; // rax
  __int64 *v4; // rbx
  __int64 v5; // r13
  unsigned __int64 v6; // r14
  __int64 v7; // r15
  char v8; // si
  char v9; // al
  char *v10; // rax
  __int64 *v11; // rdx
  int v12; // ecx
  __int64 v13; // rdi
  _BYTE *v14; // rax
  __int64 v15; // rdi
  char v16; // cl
  _BYTE *v17; // rax
  __int64 v18; // r14
  __int64 v19; // r15
  char v20; // r13
  char v21; // si
  char *v22; // rax
  char v23; // al
  __int64 v24; // rdi
  _BYTE *v25; // rax
  __int64 v26; // r15
  unsigned __int64 v27; // r14
  char v28; // si
  char v29; // al
  char *v30; // rax
  __int64 v31; // r14
  _QWORD *v32; // r15
  __int64 v33; // rax
  __int64 *v35; // [rsp+18h] [rbp-58h]
  __int64 v36[2]; // [rsp+20h] [rbp-50h] BYREF
  __int64 v37; // [rsp+30h] [rbp-40h]
  unsigned int v38; // [rsp+38h] [rbp-38h]

  result = *(unsigned int *)(a1 + 744);
  if ( (_DWORD)result )
  {
    sub_1079610(a1, (__int64)v36, 11);
    sub_107A5C0(*(unsigned int *)(a1 + 744), **(_QWORD **)(a1 + 104), 0);
    v4 = *(__int64 **)(a1 + 736);
    v5 = 10LL * *(unsigned int *)(a1 + 744);
    v35 = &v4[v5];
    if ( &v4[v5] == v4 )
      goto LABEL_33;
LABEL_3:
    v6 = *((unsigned int *)v4 + 6);
    v7 = **(_QWORD **)(a1 + 104);
    do
    {
      while ( 1 )
      {
        v8 = v6 & 0x7F;
        v9 = v6 & 0x7F | 0x80;
        v6 >>= 7;
        if ( v6 )
          v8 = v9;
        v10 = *(char **)(v7 + 32);
        if ( (unsigned __int64)v10 >= *(_QWORD *)(v7 + 24) )
          break;
        *(_QWORD *)(v7 + 32) = v10 + 1;
        *v10 = v8;
        if ( !v6 )
          goto LABEL_9;
      }
      sub_CB5D20(v7, v8);
    }
    while ( v6 );
LABEL_9:
    v11 = *(__int64 **)(a1 + 104);
    v12 = *((_DWORD *)v4 + 6);
    v13 = *v11;
    v14 = *(_BYTE **)(*v11 + 32);
    if ( (v12 & 2) != 0 )
    {
      if ( *(_QWORD *)(*v11 + 24) <= (unsigned __int64)v14 )
      {
        sub_CB5D20(v13, 0);
      }
      else
      {
        *(_QWORD *)(v13 + 32) = v14 + 1;
        *v14 = 0;
      }
      v12 = *((_DWORD *)v4 + 6);
      v11 = *(__int64 **)(a1 + 104);
    }
    if ( (v12 & 1) != 0 )
      goto LABEL_26;
    v15 = *v11;
    v16 = 66 - ((*(_BYTE *)(*(_QWORD *)(a1 + 112) + 8LL) & 1) == 0);
    v17 = *(_BYTE **)(*v11 + 32);
    if ( (unsigned __int64)v17 >= *(_QWORD *)(*v11 + 24) )
    {
      sub_CB5D20(v15, 66 - ((*(_BYTE *)(*(_QWORD *)(a1 + 112) + 8LL) & 1) == 0));
    }
    else
    {
      *(_QWORD *)(v15 + 32) = v17 + 1;
      *v17 = v16;
    }
    v18 = v4[4];
    v19 = **(_QWORD **)(a1 + 104);
    while ( 1 )
    {
      v23 = v18;
      v21 = v18 & 0x7F;
      v18 >>= 7;
      if ( !v18 )
        break;
      if ( v18 == -1 && (v20 = 0, (v23 & 0x40) != 0) )
      {
        v22 = *(char **)(v19 + 32);
        if ( (unsigned __int64)v22 >= *(_QWORD *)(v19 + 24) )
          goto LABEL_22;
LABEL_17:
        *(_QWORD *)(v19 + 32) = v22 + 1;
        *v22 = v21;
        if ( !v20 )
          goto LABEL_23;
      }
      else
      {
LABEL_15:
        v21 |= 0x80u;
        v20 = 1;
LABEL_16:
        v22 = *(char **)(v19 + 32);
        if ( (unsigned __int64)v22 < *(_QWORD *)(v19 + 24) )
          goto LABEL_17;
LABEL_22:
        sub_CB5D20(v19, v21);
        if ( !v20 )
        {
LABEL_23:
          v24 = **(_QWORD **)(a1 + 104);
          v25 = *(_BYTE **)(v24 + 32);
          if ( (unsigned __int64)v25 >= *(_QWORD *)(v24 + 24) )
          {
            sub_CB5D20(v24, 11);
          }
          else
          {
            *(_QWORD *)(v24 + 32) = v25 + 1;
            *v25 = 11;
          }
          v11 = *(__int64 **)(a1 + 104);
LABEL_26:
          v26 = *v11;
          v27 = v4[7];
          do
          {
            while ( 1 )
            {
              v28 = v27 & 0x7F;
              v29 = v27 & 0x7F | 0x80;
              v27 >>= 7;
              if ( v27 )
                v28 = v29;
              v30 = *(char **)(v26 + 32);
              if ( (unsigned __int64)v30 >= *(_QWORD *)(v26 + 24) )
                break;
              *(_QWORD *)(v26 + 32) = v30 + 1;
              *v30 = v28;
              if ( !v27 )
                goto LABEL_32;
            }
            sub_CB5D20(v26, v28);
          }
          while ( v27 );
LABEL_32:
          v31 = *v4;
          v4 += 10;
          v32 = **(_QWORD ***)(a1 + 104);
          v33 = (*(__int64 (__fastcall **)(_QWORD *))(*v32 + 80LL))(v32);
          *(_QWORD *)(v31 + 160) = v33 + v32[4] - v32[2] - v37;
          sub_CB6200(**(_QWORD **)(a1 + 104), (unsigned __int8 *)*(v4 - 4), *(v4 - 3));
          if ( v35 == v4 )
          {
LABEL_33:
            sub_107E490(
              a1,
              *(__int64 **)(a1 + 144),
              0xCCCCCCCCCCCCCCCDLL * ((__int64)(*(_QWORD *)(a1 + 152) - *(_QWORD *)(a1 + 144)) >> 3),
              v37,
              a2);
            sub_1077B30(a1, v36);
            return v38;
          }
          goto LABEL_3;
        }
      }
    }
    v20 = 0;
    if ( (v23 & 0x40) == 0 )
      goto LABEL_16;
    goto LABEL_15;
  }
  return result;
}
