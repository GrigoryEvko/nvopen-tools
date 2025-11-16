// Function: sub_2DD0740
// Address: 0x2dd0740
//
__int64 __fastcall sub_2DD0740(__int64 a1)
{
  __int64 v2; // rsi
  __int64 v3; // rsi
  unsigned __int64 *v4; // rbx
  unsigned __int64 *v5; // r12
  unsigned __int64 v6; // r13
  unsigned __int64 v7; // r8
  __int64 v8; // rbx
  __int64 v9; // rbx
  __int64 v10; // r12
  _QWORD *v11; // rdi
  __int64 v12; // rbx
  unsigned __int64 v13; // r13
  unsigned __int64 v14; // rdi
  _QWORD *v15; // r12
  __int64 (__fastcall *v16)(_QWORD *); // rax

  v2 = *(unsigned int *)(a1 + 272);
  *(_QWORD *)a1 = &unk_4A27B98;
  v3 = 16 * v2;
  sub_C7D6A0(*(_QWORD *)(a1 + 256), v3, 8);
  v4 = *(unsigned __int64 **)(a1 + 232);
  v5 = *(unsigned __int64 **)(a1 + 224);
  if ( v4 != v5 )
  {
    do
    {
      v6 = *v5;
      if ( *v5 )
      {
        sub_2DD06B0((_QWORD *)*v5);
        v3 = 72;
        j_j___libc_free_0(v6);
      }
      ++v5;
    }
    while ( v4 != v5 );
    v5 = *(unsigned __int64 **)(a1 + 224);
  }
  if ( v5 )
  {
    v3 = *(_QWORD *)(a1 + 240) - (_QWORD)v5;
    j_j___libc_free_0((unsigned __int64)v5);
  }
  v7 = *(_QWORD *)(a1 + 200);
  if ( *(_DWORD *)(a1 + 212) )
  {
    v8 = *(unsigned int *)(a1 + 208);
    if ( (_DWORD)v8 )
    {
      v9 = 8 * v8;
      v10 = 0;
      do
      {
        v11 = *(_QWORD **)(v7 + v10);
        if ( v11 != (_QWORD *)-8LL && v11 )
        {
          v3 = *v11 + 17LL;
          sub_C7D6A0((__int64)v11, v3, 8);
          v7 = *(_QWORD *)(a1 + 200);
        }
        v10 += 8;
      }
      while ( v9 != v10 );
    }
  }
  _libc_free(v7);
  v12 = *(_QWORD *)(a1 + 176);
  v13 = v12 + 8LL * *(unsigned int *)(a1 + 184);
  if ( v12 != v13 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v15 = *(_QWORD **)(v13 - 8);
        v13 -= 8LL;
        if ( v15 )
          break;
LABEL_20:
        if ( v12 == v13 )
          goto LABEL_24;
      }
      v16 = *(__int64 (__fastcall **)(_QWORD *))(*v15 + 8LL);
      if ( v16 == sub_BD9990 )
      {
        v14 = v15[1];
        *v15 = &unk_49DB390;
        if ( (_QWORD *)v14 != v15 + 3 )
          j_j___libc_free_0(v14);
        v3 = 48;
        j_j___libc_free_0((unsigned __int64)v15);
        goto LABEL_20;
      }
      ((void (__fastcall *)(_QWORD *, __int64))v16)(v15, v3);
      if ( v12 == v13 )
      {
LABEL_24:
        v13 = *(_QWORD *)(a1 + 176);
        break;
      }
    }
  }
  if ( v13 != a1 + 192 )
    _libc_free(v13);
  return sub_BB9280(a1);
}
