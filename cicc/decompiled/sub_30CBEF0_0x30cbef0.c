// Function: sub_30CBEF0
// Address: 0x30cbef0
//
_QWORD *__fastcall sub_30CBEF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int32 a5)
{
  __m128i *v6; // rdi
  _QWORD *result; // rax
  __int64 v8; // rax
  __int64 v9; // rdi
  unsigned __int64 v10; // r15
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // r13
  _QWORD *v15; // r14
  unsigned __int64 *v16; // r10
  __int64 v17; // r9
  unsigned __int64 *v18; // [rsp+8h] [rbp-58h]
  __int64 v19; // [rsp+10h] [rbp-50h]
  __int64 v20; // [rsp+10h] [rbp-50h]
  __int64 v21; // [rsp+18h] [rbp-48h]

  v6 = (__m128i *)(a1 + 40);
  v6[-2].m128i_i64[0] = a2;
  v6[-2].m128i_i64[1] = a3;
  v6[-1].m128i_i64[0] = a4;
  v6[-3].m128i_i64[1] = (__int64)&unk_4A325A8;
  v6[-1].m128i_i32[2] = a5;
  if ( (_BYTE)a5 && (_BYTE)qword_502F7C8 )
  {
    sub_30CBBD0(v6, a4);
  }
  else
  {
    *(_QWORD *)(a1 + 40) = a1 + 56;
    sub_30CA380(v6->m128i_i64, "inline", (__int64)"");
  }
  result = &qword_5041320;
  *(_QWORD *)(a1 + 72) = 0;
  if ( LODWORD(qword_5041368[8]) )
  {
    v8 = sub_22077B0(0x48u);
    v9 = v8;
    if ( v8 )
    {
      *(_QWORD *)(v8 + 64) = 0;
      *(_OWORD *)(v8 + 16) = 0;
      *(_BYTE *)(v8 + 20) = 16;
      *(_OWORD *)v8 = 0;
      *(_OWORD *)(v8 + 32) = 0;
      *(_OWORD *)(v8 + 48) = 0;
    }
    v10 = *(_QWORD *)(a1 + 72);
    *(_QWORD *)(a1 + 72) = v8;
    if ( v10 )
    {
      v11 = *(_QWORD *)(v10 + 24);
      if ( v11 )
        j_j___libc_free_0(v11);
      v12 = *(_QWORD *)v10;
      if ( *(_DWORD *)(v10 + 12) )
      {
        v13 = *(unsigned int *)(v10 + 8);
        if ( (_DWORD)v13 )
        {
          v14 = 0;
          v21 = 8 * v13;
          do
          {
            v15 = *(_QWORD **)(v12 + v14);
            if ( v15 && v15 != (_QWORD *)-8LL )
            {
              v16 = (unsigned __int64 *)v15[1];
              v17 = *v15 + 17LL;
              if ( v16 )
              {
                if ( (unsigned __int64 *)*v16 != v16 + 2 )
                {
                  v18 = (unsigned __int64 *)v15[1];
                  v19 = *v15 + 17LL;
                  _libc_free(*v16);
                  v16 = v18;
                  v17 = v19;
                }
                v20 = v17;
                j_j___libc_free_0((unsigned __int64)v16);
                v17 = v20;
              }
              sub_C7D6A0((__int64)v15, v17, 8);
              v12 = *(_QWORD *)v10;
            }
            v14 += 8;
          }
          while ( v21 != v14 );
        }
      }
      _libc_free(v12);
      j_j___libc_free_0(v10);
      v9 = *(_QWORD *)(a1 + 72);
    }
    return (_QWORD *)sub_36FCC20(v9, a2);
  }
  return result;
}
