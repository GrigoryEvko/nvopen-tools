// Function: sub_1079790
// Address: 0x1079790
//
__int64 __fastcall sub_1079790(__int64 a1, __int64 a2, __int64 a3, size_t a4)
{
  _QWORD *v8; // r15
  size_t v9; // r15
  __int64 v10; // r13
  char v11; // si
  char v12; // al
  char *v13; // rax
  _QWORD *v14; // r15
  void *v15; // rdi
  __int64 result; // rax
  _QWORD *v17; // r14
  __int64 v18; // rax
  unsigned __int64 v19; // rax
  unsigned int v20; // r14d
  __int64 v21; // r15
  int v22; // r8d
  _BYTE *v23; // rax
  int v24; // edx
  _BYTE *v25; // rax
  _BYTE *v26; // rax
  __int64 v27; // rdi
  __int64 v28; // rdx
  int v29; // [rsp+8h] [rbp-78h]
  int v30; // [rsp+8h] [rbp-78h]
  int v32; // [rsp+1Ch] [rbp-64h]
  _QWORD v33[12]; // [rsp+20h] [rbp-60h] BYREF

  sub_1079610(a1, a2, 0);
  v8 = **(_QWORD ***)(a1 + 104);
  *(_QWORD *)(a2 + 8) = (*(__int64 (__fastcall **)(_QWORD *))(*v8 + 80LL))(v8) + v8[4] - v8[2];
  if ( a4 == 10 && *(_QWORD *)a3 == 0x61676E616C635F5FLL && *(_WORD *)(a3 + 8) == 29811 )
  {
    memset(&v33[1], 0, 40);
    v33[0] = &unk_49DD308;
    sub_CB5D20((__int64)v33, 10);
    v17 = **(_QWORD ***)(a1 + 104);
    v18 = (*(__int64 (__fastcall **)(_QWORD *))(*v17 + 80LL))(v17);
    v19 = ((v17[4] - v17[2] + v18 + 14) & 0xFFFFFFFFFFFFFFFCLL) - (v17[4] - v17[2] + v18 + 11);
    v20 = v19 + 1;
    v29 = v19;
    v32 = v19;
    v21 = **(_QWORD **)(a1 + 104);
    v22 = (unsigned int)(v19 + 1) < 2 ? -246 : -118;
    v23 = *(_BYTE **)(v21 + 32);
    if ( (unsigned __int64)v23 >= *(_QWORD *)(v21 + 24) )
    {
      sub_CB5D20(**(_QWORD **)(a1 + 104), v22);
    }
    else
    {
      *(_QWORD *)(v21 + 32) = v23 + 1;
      *v23 = v22;
    }
    if ( v20 > 1 )
    {
      v24 = 1;
      if ( v29 != 1 )
      {
        do
        {
          while ( 1 )
          {
            v25 = *(_BYTE **)(v21 + 32);
            if ( (unsigned __int64)v25 >= *(_QWORD *)(v21 + 24) )
              break;
            ++v24;
            *(_QWORD *)(v21 + 32) = v25 + 1;
            *v25 = 0x80;
            if ( v32 == v24 )
              goto LABEL_23;
          }
          v30 = v24;
          sub_CB5D20(v21, 128);
          v24 = v30 + 1;
        }
        while ( v32 != v30 + 1 );
      }
LABEL_23:
      v26 = *(_BYTE **)(v21 + 32);
      if ( (unsigned __int64)v26 >= *(_QWORD *)(v21 + 24) )
      {
        sub_CB5D20(v21, 0);
      }
      else
      {
        *(_QWORD *)(v21 + 32) = v26 + 1;
        *v26 = 0;
      }
    }
    v27 = **(_QWORD **)(a1 + 104);
    v28 = *(_QWORD *)(v27 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(v27 + 24) - v28) <= 9 )
    {
      sub_CB6200(v27, (unsigned __int8 *)a3, 0xAu);
    }
    else
    {
      *(_QWORD *)v28 = *(_QWORD *)a3;
      *(_WORD *)(v28 + 8) = *(_WORD *)(a3 + 8);
      *(_QWORD *)(v27 + 32) += 10LL;
    }
    sub_CB58D0((__int64)v33);
    v14 = **(_QWORD ***)(a1 + 104);
  }
  else
  {
    v9 = a4;
    v10 = **(_QWORD **)(a1 + 104);
    do
    {
      while ( 1 )
      {
        v11 = v9 & 0x7F;
        v12 = v9 & 0x7F | 0x80;
        v9 >>= 7;
        if ( v9 )
          v11 = v12;
        v13 = *(char **)(v10 + 32);
        if ( (unsigned __int64)v13 >= *(_QWORD *)(v10 + 24) )
          break;
        *(_QWORD *)(v10 + 32) = v13 + 1;
        *v13 = v11;
        if ( !v9 )
          goto LABEL_9;
      }
      sub_CB5D20(v10, v11);
    }
    while ( v9 );
LABEL_9:
    v14 = **(_QWORD ***)(a1 + 104);
    v15 = (void *)v14[4];
    if ( v14[3] - (_QWORD)v15 < a4 )
    {
      sub_CB6200(**(_QWORD **)(a1 + 104), (unsigned __int8 *)a3, a4);
      v14 = **(_QWORD ***)(a1 + 104);
    }
    else if ( a4 )
    {
      memcpy(v15, (const void *)a3, a4);
      v14[4] += a4;
      v14 = **(_QWORD ***)(a1 + 104);
    }
  }
  result = (*(__int64 (__fastcall **)(_QWORD *))(*v14 + 80LL))(v14) + v14[4] - v14[2];
  *(_QWORD *)(a2 + 16) = result;
  return result;
}
