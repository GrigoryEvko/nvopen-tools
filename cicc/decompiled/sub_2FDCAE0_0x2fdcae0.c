// Function: sub_2FDCAE0
// Address: 0x2fdcae0
//
void __fastcall sub_2FDCAE0(__int64 a1, unsigned int a2, unsigned int a3, __int64 *a4)
{
  __int64 v5; // r15
  __int64 v7; // rax
  __int64 *v8; // r8
  __int64 v9; // r9
  const __m128i *v10; // rdi
  unsigned int v11; // eax
  __int64 v12; // rdx
  unsigned int v13; // r14d
  __int64 v14; // rcx
  __int64 v15; // rax
  unsigned int v16; // [rsp+4h] [rbp-11Ch]
  __int64 v18; // [rsp+8h] [rbp-118h]
  const __m128i *v19; // [rsp+10h] [rbp-110h] BYREF
  __int64 v20; // [rsp+18h] [rbp-108h]
  _BYTE v21[256]; // [rsp+20h] [rbp-100h] BYREF

  v5 = 40LL * a2;
  if ( (*(_WORD *)(*(_QWORD *)(a1 + 32) + v5 + 2) & 0xFF0) != 0 )
  {
    v11 = sub_2E89F40(a1, a2);
    v12 = a3;
    v13 = v11;
    v14 = v5 + *(_QWORD *)(a1 + 32);
    if ( !*(_BYTE *)v14 && (*(_WORD *)(v14 + 2) & 0xFF0) != 0 )
    {
      v16 = a3;
      v18 = v5 + *(_QWORD *)(a1 + 32);
      v15 = *(_QWORD *)(a1 + 32) + 40LL * (unsigned int)sub_2E89F40(a1, a2);
      *(_WORD *)(v15 + 2) &= 0xF00Fu;
      *(_WORD *)(v18 + 2) &= 0xF00Fu;
      v12 = v16;
    }
    sub_2FDCAE0(a1, v13, v12, a4);
  }
  v20 = 0x500000000LL;
  v7 = *a4;
  v19 = (const __m128i *)v21;
  (*(void (__fastcall **)(__int64 *, const __m128i **))(v7 + 1528))(a4, &v19);
  sub_2E8A650(a1, a2);
  sub_2E91190(a1, v5 + *(_QWORD *)(a1 + 32), v19, (unsigned int)v20, v8, v9);
  v10 = v19;
  *(_QWORD *)(*(_QWORD *)(a1 + 32) + 40LL * (a2 - 1) + 24) = (8 * (_DWORD)v20) & 0x8000FFF9 | 0x40006;
  if ( v10 != (const __m128i *)v21 )
    _libc_free((unsigned __int64)v10);
}
