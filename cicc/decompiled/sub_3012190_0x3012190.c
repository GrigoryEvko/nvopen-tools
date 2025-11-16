// Function: sub_3012190
// Address: 0x3012190
//
_QWORD *__fastcall sub_3012190(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  unsigned __int64 v7; // rax
  __int64 v8; // rdx
  unsigned __int64 v9; // rcx
  unsigned __int64 v10; // r12
  __int64 v11; // r12
  _QWORD *result; // rax
  __int64 v13; // rax
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rax

  v6 = sub_AA4FF0(a1);
  if ( !v6 )
    goto LABEL_17;
  v7 = (unsigned int)*(unsigned __int8 *)(v6 - 24) - 39;
  if ( (unsigned int)v7 > 0x38 )
    goto LABEL_4;
  v8 = 0x100060000000001LL;
  if ( !_bittest64(&v8, v7) )
    goto LABEL_4;
  v13 = sub_AA4FF0(a1);
  if ( !v13 )
    goto LABEL_17;
  if ( (unsigned int)*(unsigned __int8 *)(v13 - 24) - 30 > 0xA )
  {
LABEL_4:
    v9 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v9 == a1 + 48 )
    {
      v10 = 0;
LABEL_8:
      v11 = v10 + 24;
      result = sub_BD2C40(80, unk_3F10A10);
      if ( result )
        return (_QWORD *)sub_B4D460((__int64)result, a2, a3, v11, 0);
      return result;
    }
    if ( v9 )
    {
      v10 = v9 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v9 - 24) - 30 >= 0xB )
        v10 = 0;
      goto LABEL_8;
    }
LABEL_17:
    BUG();
  }
  v16 = *(unsigned int *)(a4 + 8);
  if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
  {
    sub_C8D5F0(a4, (const void *)(a4 + 16), v16 + 1, 0x10u, v14, v15);
    v16 = *(unsigned int *)(a4 + 8);
  }
  result = (_QWORD *)(*(_QWORD *)a4 + 16 * v16);
  *result = a1;
  result[1] = a2;
  ++*(_DWORD *)(a4 + 8);
  return result;
}
