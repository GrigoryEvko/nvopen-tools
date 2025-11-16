// Function: sub_26449B0
// Address: 0x26449b0
//
__int64 __fastcall sub_26449B0(_QWORD *a1, __m128i *a2, __m128i *a3, size_t a4, __m128i *a5, size_t a6)
{
  __int64 v6; // r15
  __int64 v7; // rax
  size_t v8; // r13
  __int64 v9; // rcx
  __m128i *v10; // r14
  __int64 v11; // rbx
  __int64 v12; // r12
  int v13; // eax
  int v14; // eax
  bool v15; // zf
  __int64 result; // rax
  __int64 *v17; // rbx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int16 v21; // [rsp+4h] [rbp-DCh]
  __int64 v22; // [rsp+8h] [rbp-D8h]
  int v23; // [rsp+1Ch] [rbp-C4h] BYREF
  __m128i v24; // [rsp+20h] [rbp-C0h] BYREF
  __m128i *v25; // [rsp+30h] [rbp-B0h]
  size_t v26; // [rsp+38h] [rbp-A8h]
  __int16 v27; // [rsp+40h] [rbp-A0h]
  __m128i v28[2]; // [rsp+50h] [rbp-90h] BYREF
  char v29; // [rsp+70h] [rbp-70h]
  char v30; // [rsp+71h] [rbp-6Fh]
  __m128i v31[6]; // [rsp+80h] [rbp-60h] BYREF

  v6 = (__int64)a1;
  v7 = a1[21];
  v21 = (__int16)a2;
  v23 = 0;
  if ( *(_QWORD *)(v7 + 32) )
  {
    a4 = a6;
    a3 = a5;
  }
  v8 = a4;
  v9 = *((unsigned int *)a1 + 46);
  v10 = a3;
  if ( *((_DWORD *)a1 + 46) )
  {
    v11 = a1[22];
    v12 = 0;
    while ( 1 )
    {
      if ( v8 == *(_QWORD *)(v11 + 8) )
      {
        v22 = v9;
        if ( !v8 )
          break;
        a1 = *(_QWORD **)v11;
        a2 = v10;
        v13 = memcmp(*(const void **)v11, v10, v8);
        v9 = v22;
        if ( !v13 )
          break;
      }
      ++v12;
      v11 += 48;
      if ( v9 == v12 )
        goto LABEL_13;
    }
    v14 = *(_DWORD *)(v11 + 40);
    v23 = v14;
  }
  else
  {
LABEL_13:
    v30 = 1;
    v17 = sub_CEADF0();
    v29 = 3;
    v28[0].m128i_i64[0] = (__int64)"'!";
    v27 = 1283;
    v24.m128i_i64[0] = (__int64)"Cannot find option named '";
    v25 = v10;
    v26 = v8;
    sub_9C6370(v31, &v24, v28, v18, v19, v20);
    a2 = v31;
    a1 = (_QWORD *)v6;
    result = sub_C53280(v6, (__int64)v31, 0, 0, (__int64)v17);
    if ( (_BYTE)result )
      return result;
    v14 = v23;
  }
  *(_DWORD *)(v6 + 136) = v14;
  v15 = *(_QWORD *)(v6 + 592) == 0;
  *(_WORD *)(v6 + 14) = v21;
  if ( v15 )
    sub_4263D6(a1, a2, a3);
  (*(void (__fastcall **)(__int64, int *, __m128i *, __int64))(v6 + 600))(v6 + 576, &v23, a3, v9);
  return 0;
}
