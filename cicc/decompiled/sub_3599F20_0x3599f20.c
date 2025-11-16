// Function: sub_3599F20
// Address: 0x3599f20
//
void (*__fastcall sub_3599F20(__int64 a1))()
{
  __int64 v2; // rdx
  int v3; // ecx
  __int64 v4; // r14
  __int64 v5; // rdi
  void (__fastcall *v6)(__int64, __int64, __int64, __int64, _BYTE *, _QWORD, __int64 *); // rax
  __int64 v7; // r12
  __int64 v8; // rdi
  __int64 v9; // r15
  __int64 v10; // rax
  __int16 v11; // ax
  char v12; // bl
  __int64 v13; // rax
  __int64 v14; // r15
  __int64 v15; // rbx
  __int64 v16; // r15
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // r12
  __int64 i; // rbx
  __int64 v21; // rdi
  void (*result)(); // rax
  __int64 v23; // [rsp+10h] [rbp-110h]
  __int64 *v24; // [rsp+20h] [rbp-100h]
  char v25; // [rsp+2Bh] [rbp-F5h]
  char v26; // [rsp+2Bh] [rbp-F5h]
  unsigned int v27; // [rsp+2Ch] [rbp-F4h]
  __int64 v28; // [rsp+38h] [rbp-E8h] BYREF
  _BYTE *v29; // [rsp+40h] [rbp-E0h] BYREF
  __int64 v30; // [rsp+48h] [rbp-D8h]
  _BYTE v31[208]; // [rsp+50h] [rbp-D0h] BYREF

  v2 = *(_QWORD *)(a1 + 64);
  v3 = *(_DWORD *)(*(_QWORD *)a1 + 96LL);
  v27 = v3 - 1;
  v4 = v2 + 8LL * *(unsigned int *)(a1 + 72);
  if ( v4 == v2 )
    goto LABEL_35;
  v25 = 0;
  v24 = (__int64 *)(*(_QWORD *)(a1 + 112) + 8LL * *(unsigned int *)(a1 + 120) - 8);
  do
  {
    v7 = *(_QWORD *)(v4 - 8);
    v8 = *(_QWORD *)(a1 + 32);
    v9 = **(_QWORD **)(v7 + 112);
    v10 = *v24;
    v29 = v31;
    v30 = 0x400000000LL;
    v23 = v10;
    (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v8 + 360LL))(v8, v7, 0);
    v11 = (*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64, _BYTE **))(**(_QWORD **)(a1 + 528) + 32LL))(
            *(_QWORD *)(a1 + 528),
            v27,
            v7,
            &v29);
    v12 = HIBYTE(v11);
    if ( HIBYTE(v11) )
    {
      if ( (_BYTE)v11 )
      {
        sub_2E33650(v7, v23);
        v18 = sub_2E311E0(v23);
        v19 = *(_QWORD *)(v23 + 56);
        for ( i = v18; v19 != i; v19 = *(_QWORD *)(v19 + 8) )
        {
          sub_2E8A650(v19, 4u);
          sub_2E8A650(v19, 3u);
          if ( !v19 )
LABEL_37:
            BUG();
          if ( (*(_BYTE *)v19 & 4) == 0 && (*(_BYTE *)(v19 + 44) & 8) != 0 )
          {
            do
              v19 = *(_QWORD *)(v19 + 8);
            while ( (*(_BYTE *)(v19 + 44) & 8) != 0 );
          }
        }
      }
      else
      {
        sub_2E33650(v7, v9);
        v13 = sub_2E311E0(v9);
        v14 = *(_QWORD *)(v9 + 56);
        if ( v14 != v13 )
        {
          v26 = v12;
          v15 = v14;
          v16 = v13;
          do
          {
            while ( 1 )
            {
              sub_2E8A650(v15, 2u);
              sub_2E8A650(v15, 1u);
              if ( !v15 )
                goto LABEL_37;
              if ( (*(_BYTE *)v15 & 4) == 0 )
                break;
              v15 = *(_QWORD *)(v15 + 8);
              if ( v16 == v15 )
                goto LABEL_16;
            }
            while ( (*(_BYTE *)(v15 + 44) & 8) != 0 )
              v15 = *(_QWORD *)(v15 + 8);
            v15 = *(_QWORD *)(v15 + 8);
          }
          while ( v16 != v15 );
LABEL_16:
          v12 = v26;
        }
        v17 = *(_QWORD *)(a1 + 32);
        v28 = 0;
        (*(void (__fastcall **)(__int64, __int64, __int64, _QWORD, _QWORD, _QWORD, __int64 *, _QWORD))(*(_QWORD *)v17 + 368LL))(
          v17,
          v7,
          v23,
          0,
          0,
          0,
          &v28,
          0);
        if ( v28 )
          sub_B91220((__int64)&v28, v28);
        v25 = v12;
      }
    }
    else
    {
      v5 = *(_QWORD *)(a1 + 32);
      v6 = *(void (__fastcall **)(__int64, __int64, __int64, __int64, _BYTE *, _QWORD, __int64 *))(*(_QWORD *)v5 + 368LL);
      v28 = 0;
      v6(v5, v7, v23, v9, v29, (unsigned int)v30, &v28);
      if ( v28 )
        sub_B91220((__int64)&v28, v28);
    }
    if ( v29 != v31 )
      _libc_free((unsigned __int64)v29);
    --v27;
    v4 -= 8;
    --v24;
  }
  while ( *(_QWORD *)(a1 + 64) != v4 );
  if ( !v25 )
  {
    v3 = *(_DWORD *)(*(_QWORD *)a1 + 96LL);
LABEL_35:
    (*(void (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a1 + 528) + 48LL))(
      *(_QWORD *)(a1 + 528),
      (unsigned int)(1 - v3));
    return (void (*)())(*(__int64 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a1 + 528) + 56LL))(
                         *(_QWORD *)(a1 + 528),
                         *(_QWORD *)(*(_QWORD *)(a1 + 64) + 8LL * *(unsigned int *)(a1 + 72) - 8));
  }
  v21 = *(_QWORD *)(a1 + 528);
  result = *(void (**)())(*(_QWORD *)v21 + 64LL);
  if ( result != nullsub_1682 )
    return (void (*)())((__int64 (__fastcall *)(__int64, _QWORD))result)(v21, 0);
  return result;
}
