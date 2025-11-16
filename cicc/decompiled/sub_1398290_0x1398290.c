// Function: sub_1398290
// Address: 0x1398290
//
__int64 __fastcall sub_1398290(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rdi
  _QWORD *v6; // r12
  void (__fastcall *v7)(__int64, __int64); // rbx
  __int64 v8; // rdx
  __int64 v9; // r8
  __int64 v10; // rdi
  __m128i *v12; // rdx
  __m128i si128; // xmm0

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_13:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F98A8D )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_13;
  }
  v5 = *(_QWORD *)(v3 + 8);
  v6 = (_QWORD *)(*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v5 + 104LL))(v5, &unk_4F98A8D);
  v7 = *(void (__fastcall **)(__int64, __int64))(*v6 + 40LL);
  v9 = sub_16E8CB0(v5, &unk_4F98A8D, v8);
  if ( v7 == sub_1398210 )
  {
    v10 = v6[20];
    if ( v10 )
    {
      sub_1397F00(v10, v9);
    }
    else
    {
      v12 = *(__m128i **)(v9 + 24);
      if ( *(_QWORD *)(v9 + 16) - (_QWORD)v12 <= 0x1Du )
      {
        sub_16E7EE0(v9, "No call graph has been built!\n", 30);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_3F70830);
        qmemcpy(&v12[1], "s been built!\n", 14);
        *v12 = si128;
        *(_QWORD *)(v9 + 24) += 30LL;
      }
    }
    return 0;
  }
  else
  {
    ((void (__fastcall *)(_QWORD *, __int64, __int64))v7)(v6, v9, a2);
    return 0;
  }
}
