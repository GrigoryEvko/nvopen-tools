// Function: sub_89A380
// Address: 0x89a380
//
__int64 __fastcall sub_89A380(int a1, __m128i *a2, __m128i **a3, __m128i **a4, int *a5, _DWORD *a6)
{
  __m128i *v8; // rbx
  __int64 v9; // rax
  __m128i *v10; // r12
  __int64 result; // rax
  int *v12; // r8
  _DWORD *v13; // r9
  int v14; // esi
  int v15; // [rsp+Ch] [rbp-44h]

  v8 = a2;
  if ( dword_4F04C44 != -1
    || (v9 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v9 + 6) & 6) != 0)
    || *(_BYTE *)(v9 + 4) == 12 )
  {
    v10 = sub_72F240(a2);
    sub_88E6E0(v10->m128i_i64, 1u);
    v15 = sub_89A370(v10->m128i_i64);
    sub_88E6E0(a2->m128i_i64, 0);
    result = sub_89A370(a2->m128i_i64);
    v12 = a5;
    v13 = a6;
    if ( !v10 )
    {
      v15 = result;
      v10 = a2;
    }
    v14 = 1;
  }
  else
  {
    v10 = a2;
    sub_88E6E0(a2->m128i_i64, 0);
    result = sub_89A370(a2->m128i_i64);
    v12 = a5;
    v13 = a6;
    v14 = 0;
    v15 = result;
  }
  if ( a1 )
  {
    v8 = v10;
  }
  else if ( dword_4F04C44 != -1 && v15 && (_DWORD)result )
  {
    v8 = v10;
  }
  *a3 = v10;
  *a4 = v8;
  *v12 = v14;
  *v13 = result;
  return result;
}
