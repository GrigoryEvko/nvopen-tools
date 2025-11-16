// Function: sub_2641EB0
// Address: 0x2641eb0
//
__int64 __fastcall sub_2641EB0(__int64 a1, const __m128i *a2, __int64 a3)
{
  __int64 v5; // r12
  __int64 v6; // rdx
  __int64 v7; // rax
  int v8; // esi
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rbx
  _QWORD *v13; // rcx
  char v14; // di
  unsigned __int64 v16; // rax

  v5 = sub_22077B0(0x60u);
  v6 = v5 + 56;
  *(__m128i *)(v5 + 32) = _mm_loadu_si128(a2);
  v7 = *(_QWORD *)(a3 + 16);
  if ( v7 )
  {
    v8 = *(_DWORD *)(a3 + 8);
    *(_QWORD *)(v5 + 64) = v7;
    *(_DWORD *)(v5 + 56) = v8;
    *(_QWORD *)(v5 + 72) = *(_QWORD *)(a3 + 24);
    *(_QWORD *)(v5 + 80) = *(_QWORD *)(a3 + 32);
    *(_QWORD *)(v7 + 8) = v6;
    v9 = *(_QWORD *)(a3 + 40);
    *(_QWORD *)(a3 + 16) = 0;
    *(_QWORD *)(v5 + 88) = v9;
    *(_QWORD *)(a3 + 24) = a3 + 8;
    *(_QWORD *)(a3 + 32) = a3 + 8;
    *(_QWORD *)(a3 + 40) = 0;
  }
  else
  {
    *(_DWORD *)(v5 + 56) = 0;
    *(_QWORD *)(v5 + 64) = 0;
    *(_QWORD *)(v5 + 72) = v6;
    *(_QWORD *)(v5 + 80) = v6;
    *(_QWORD *)(v5 + 88) = 0;
  }
  v10 = sub_263E300(a1, (unsigned __int64 *)(v5 + 32));
  v12 = v10;
  if ( v11 )
  {
    v13 = (_QWORD *)(a1 + 8);
    v14 = 1;
    if ( !v10 && (_QWORD *)v11 != v13 )
    {
      v16 = *(_QWORD *)(v11 + 32);
      if ( *(_QWORD *)(v5 + 32) >= v16 )
      {
        v14 = 0;
        if ( *(_QWORD *)(v5 + 32) == v16 )
          v14 = *(_DWORD *)(v5 + 40) < *(_DWORD *)(v11 + 40);
      }
    }
    sub_220F040(v14, v5, (_QWORD *)v11, v13);
    ++*(_QWORD *)(a1 + 40);
    return v5;
  }
  else
  {
    sub_2641CE0(*(_QWORD *)(v5 + 64));
    j_j___libc_free_0(v5);
    return v12;
  }
}
