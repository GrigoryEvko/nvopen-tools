// Function: sub_D73680
// Address: 0xd73680
//
void __fastcall sub_D73680(_QWORD *a1, __int64 a2, char a3)
{
  unsigned int v6; // eax
  __int64 v7; // rdx
  _QWORD *v8; // r13
  _QWORD *i; // rbx
  __int64 v10; // rdi
  __int64 v11; // rsi
  __int64 v12; // r13
  __int64 v13; // r15
  __int64 v14; // r14
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rbx
  __int64 v20; // r13
  __int64 v21; // rcx
  unsigned int v22; // eax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // [rsp+0h] [rbp-D0h] BYREF
  char *v27; // [rsp+8h] [rbp-C8h]
  __int64 v28; // [rsp+10h] [rbp-C0h]
  int v29; // [rsp+18h] [rbp-B8h]
  char v30; // [rsp+1Ch] [rbp-B4h]
  char v31; // [rsp+20h] [rbp-B0h] BYREF

  ++a1[51];
  if ( *((_BYTE *)a1 + 436) )
    goto LABEL_6;
  v6 = 4 * (*((_DWORD *)a1 + 107) - *((_DWORD *)a1 + 108));
  v7 = *((unsigned int *)a1 + 106);
  if ( v6 < 0x20 )
    v6 = 32;
  if ( v6 >= (unsigned int)v7 )
  {
    memset((void *)a1[52], -1, 8 * v7);
LABEL_6:
    *(_QWORD *)((char *)a1 + 428) = 0;
    goto LABEL_7;
  }
  sub_C8C990((__int64)(a1 + 51), a2);
LABEL_7:
  v8 = (_QWORD *)a1[1];
  for ( i = &v8[3 * *((unsigned int *)a1 + 4)]; v8 != i; sub_D68D70(i) )
    i -= 3;
  *((_DWORD *)a1 + 4) = 0;
  v10 = a2 - 64;
  v11 = sub_D735C0(a1, a2);
  if ( *(_BYTE *)a2 == 26 )
    v10 = a2 - 32;
  sub_AC2B30(v10, v11);
  if ( a3 )
  {
    v12 = *((unsigned int *)a1 + 4);
    if ( (_DWORD)v12 )
    {
      v13 = *(_QWORD *)(a2 + 64);
      v14 = *a1;
      v26 = 0;
      v27 = &v31;
      v15 = v13;
      v28 = 16;
      v29 = 0;
      v30 = 1;
      v16 = sub_D68C20(v14, v13);
      if ( v16 )
      {
        v17 = *(_QWORD *)(v16 + 8);
        if ( !v17 )
          BUG();
        v18 = v17 - 48;
        if ( *(_BYTE *)(v17 - 48) == 27 )
          v18 = *(_QWORD *)(v17 - 112);
        v15 = v13;
        sub_D68C90(v14, v13, v18, (__int64)&v26);
        v19 = a1[1];
        v20 = v19 + 24LL * *((unsigned int *)a1 + 4);
        if ( v20 == v19 )
          goto LABEL_29;
      }
      else
      {
        v19 = a1[1];
        v20 = v19 + 24 * v12;
      }
      do
      {
        v23 = *(_QWORD *)(v19 + 16);
        if ( v23 )
        {
          v24 = *(_QWORD *)(v23 + 64);
          v25 = *(_QWORD *)(*a1 + 8LL);
          if ( v24 )
          {
            v21 = (unsigned int)(*(_DWORD *)(v24 + 44) + 1);
            v22 = *(_DWORD *)(v24 + 44) + 1;
          }
          else
          {
            v21 = 0;
            v22 = 0;
          }
          v15 = 0;
          if ( v22 < *(_DWORD *)(v25 + 32) )
            v15 = *(_QWORD *)(*(_QWORD *)(v25 + 24) + 8 * v21);
          sub_103C0D0(*a1, v15, 0, &v26, 1, 1);
        }
        v19 += 24;
      }
      while ( v19 != v20 );
LABEL_29:
      if ( !v30 )
        _libc_free(v27, v15);
    }
  }
}
