// Function: sub_3449670
// Address: 0x3449670
//
__int64 __fastcall sub_3449670(
        unsigned int *a1,
        __int64 a2,
        __int64 a3,
        _QWORD **a4,
        _QWORD **a5,
        unsigned int a6,
        __m128i a7)
{
  unsigned __int16 *v10; // rdx
  int v11; // eax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  unsigned __int64 v15; // rdx
  __int64 result; // rax
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // [rsp+0h] [rbp-80h]
  __int64 v21; // [rsp+10h] [rbp-70h]
  __int16 v22; // [rsp+20h] [rbp-60h] BYREF
  __int64 v23; // [rsp+28h] [rbp-58h]
  unsigned __int64 v24; // [rsp+30h] [rbp-50h] BYREF
  __int64 v25; // [rsp+38h] [rbp-48h]

  v10 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3);
  v11 = *v10;
  v12 = *((_QWORD *)v10 + 1);
  v22 = v11;
  v23 = v12;
  if ( (_WORD)v11 )
  {
    if ( (unsigned __int16)(v11 - 17) > 0xD3u )
    {
      LOWORD(v24) = v11;
      v25 = v12;
LABEL_4:
      if ( (_WORD)v11 == 1 || (unsigned __int16)(v11 - 504) <= 7u )
        BUG();
      v13 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v11 - 16];
      LODWORD(v25) = v13;
      if ( (unsigned int)v13 > 0x40 )
        goto LABEL_7;
      goto LABEL_11;
    }
    LOWORD(v11) = word_4456580[v11 - 1];
    v14 = 0;
  }
  else
  {
    v19 = v12;
    if ( !sub_30070B0((__int64)&v22) )
    {
      v25 = v19;
      LOWORD(v24) = 0;
      goto LABEL_10;
    }
    LOWORD(v11) = sub_3009970((__int64)&v22, a2, v19, v17, v18);
  }
  LOWORD(v24) = v11;
  v25 = v14;
  if ( (_WORD)v11 )
    goto LABEL_4;
LABEL_10:
  LODWORD(v13) = sub_3007260((__int64)&v24);
  LODWORD(v25) = v13;
  if ( (unsigned int)v13 > 0x40 )
  {
LABEL_7:
    sub_C43690((__int64)&v24, -1, 1);
    goto LABEL_14;
  }
LABEL_11:
  v15 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v13;
  if ( !(_DWORD)v13 )
    v15 = 0;
  v24 = v15;
LABEL_14:
  result = sub_3447D70(a1, a2, a3, (__int64)&v24, a4, a5, a7, a6);
  if ( (unsigned int)v25 > 0x40 )
  {
    if ( v24 )
    {
      v21 = result;
      j_j___libc_free_0_0(v24);
      return v21;
    }
  }
  return result;
}
