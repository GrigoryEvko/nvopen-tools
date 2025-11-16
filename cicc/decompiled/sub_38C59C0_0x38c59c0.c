// Function: sub_38C59C0
// Address: 0x38c59c0
//
__int64 __fastcall sub_38C59C0(__int64 a1, _QWORD *a2, __int64 a3, _QWORD *a4, __int64 a5)
{
  __int64 v9; // r8
  _QWORD *v10; // rsi
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 *v13; // r13
  __int64 v14; // rsi
  __int64 v15; // rsi
  __int64 result; // rax
  int v17; // r13d
  __int64 v18; // r13
  __int64 v19; // r15
  _QWORD *v20; // rdx
  __int64 v21; // rcx
  __int64 *v22; // [rsp+8h] [rbp-38h]

  (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a2 + 424LL))(a2, 1, 1);
  sub_38DCDD0(a2, 1);
  sub_38DCDD0(a2, (-(__int64)(*(_BYTE *)(a3 + 72) == 0) & 0xFFFFFFFFFFFFFFE9LL) + 31);
  sub_38DCDD0(a2, *(unsigned int *)(a1 + 16) + 1LL);
  v9 = *(_QWORD *)(a1 + 392);
  if ( v9 )
  {
    v10 = *(_QWORD **)(a1 + 384);
    if ( !*(_BYTE *)(a3 + 72) )
    {
LABEL_3:
      (*(void (__fastcall **)(_QWORD *, _QWORD *, __int64))(*a2 + 400LL))(a2, v10, v9);
      (*(void (__fastcall **)(_QWORD *, void *, __int64))(*a2 + 400LL))(a2, &unk_452DFBC, 1);
      v11 = *(_QWORD *)(a1 + 8);
      v12 = 32LL * *(unsigned int *)(a1 + 16);
      v22 = (__int64 *)(v11 + v12);
      if ( v11 != v11 + v12 )
      {
        v13 = *(__int64 **)(a1 + 8);
        do
        {
          v14 = *v13;
          v13 += 4;
          (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*a2 + 400LL))(a2, v14, *(v13 - 3));
          (*(void (__fastcall **)(_QWORD *, void *, __int64))(*a2 + 400LL))(a2, &unk_452DFBC, 1);
        }
        while ( v22 != v13 );
      }
      goto LABEL_6;
    }
  }
  else
  {
    v10 = a4;
    v9 = a5;
    if ( !*(_BYTE *)(a3 + 72) )
      goto LABEL_3;
  }
  sub_38C5650(a3, a2, v10, v9);
  v18 = *(_QWORD *)(a1 + 8);
  v19 = v18 + 32LL * *(unsigned int *)(a1 + 16);
  while ( v19 != v18 )
  {
    v20 = *(_QWORD **)v18;
    v21 = *(_QWORD *)(v18 + 8);
    v18 += 32;
    sub_38C5650(a3, a2, v20, v21);
  }
LABEL_6:
  (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a2 + 424LL))(
    a2,
    2 - ((*(_BYTE *)(a1 + 489) == 0) - 1LL) - ((*(_BYTE *)(a1 + 488) == 0) - 1LL),
    1);
  sub_38DCDD0(a2, 1);
  sub_38DCDD0(a2, (-(__int64)(*(_BYTE *)(a3 + 72) == 0) & 0xFFFFFFFFFFFFFFE9LL) + 31);
  sub_38DCDD0(a2, 2);
  sub_38DCDD0(a2, 15);
  if ( *(_BYTE *)(a1 + 489) )
  {
    sub_38DCDD0(a2, 5);
    sub_38DCDD0(a2, 30);
  }
  if ( *(_BYTE *)(a1 + 488) )
  {
    sub_38DCDD0(a2, 8193);
    sub_38DCDD0(a2, (-(__int64)(*(_BYTE *)(a3 + 72) == 0) & 0xFFFFFFFFFFFFFFE9LL) + 31);
  }
  sub_38DCDD0(a2, *(unsigned int *)(a1 + 128));
  v15 = a1 + 416;
  if ( !*(_QWORD *)(a1 + 424) )
    v15 = *(_QWORD *)(a1 + 120) + 72LL;
  result = sub_38C5720(a2, v15, *(_BYTE *)(a1 + 489), *(_BYTE *)(a1 + 488), a3);
  if ( *(_DWORD *)(a1 + 128) > 1u )
  {
    v17 = 1;
    result = 1;
    do
    {
      sub_38C5720(a2, *(_QWORD *)(a1 + 120) + 72 * result, *(_BYTE *)(a1 + 489), *(_BYTE *)(a1 + 488), a3);
      result = (unsigned int)(v17 + 1);
      v17 = result;
    }
    while ( (unsigned int)result < *(_DWORD *)(a1 + 128) );
  }
  return result;
}
