// Function: sub_869530
// Address: 0x869530
//
__int64 __fastcall sub_869530(__int64 a1, __int64 a2, const __m128i *a3, __int64 *a4, __int16 a5, __int64 a6, int *a7)
{
  unsigned int v8; // r8d
  __int64 v9; // rax
  __m128i *v12; // r15
  __int64 v13; // rdx
  const __m128i *v14; // r14
  _QWORD *v15; // rcx
  _BOOL4 v16; // edx
  __int64 v17; // [rsp-10h] [rbp-40h]

  *a7 = 0;
  if ( (a5 & 0x2000) == 0 && a1 )
  {
    v12 = sub_72F240(a3);
    *a7 = 0;
    v14 = sub_8680C0(a1, a2, (__int64)v12, 1u, 0, 0, a6, a7);
    if ( v14 )
    {
      v9 = sub_85B260(v17, a2, v13);
      *(_WORD *)(v9 + 40) = 1;
      v15 = qword_4F04C18;
      *(_QWORD *)(v9 + 8) = a1;
      *(_QWORD *)(v9 + 16) = v14;
      *(_QWORD *)v9 = v15;
      qword_4F04C18 = (_QWORD *)v9;
      v16 = 0;
      if ( a6 )
        v16 = *(_DWORD *)(a6 + 80) != 0;
      *(_DWORD *)(v9 + 48) = v16;
      v8 = 1;
      *(_QWORD *)(v9 + 32) = v12;
    }
    else
    {
      sub_725130(v12->m128i_i64);
      v9 = 0;
      v8 = 0;
    }
  }
  else
  {
    v8 = 1;
    v9 = 0;
  }
  *a4 = v9;
  return v8;
}
