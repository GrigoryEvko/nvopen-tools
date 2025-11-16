// Function: sub_CC80F0
// Address: 0xcc80f0
//
__int64 *__fastcall sub_CC80F0(__int64 *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v6; // r15
  __int64 v7; // rdx
  __int64 v8; // r14
  unsigned __int64 v9; // rax
  int v10; // edx
  int v11; // rdx^4
  unsigned int v12; // r14d
  unsigned int v13; // edx
  unsigned int v14; // esi
  unsigned int v15; // ecx

  if ( *(_DWORD *)(a2 + 40) == 1
    && ((v6 = sub_CC78E0(a2), v8 = v7, v9 = sub_CC78E0(a3), (unsigned int)v6 > (unsigned int)v9)
     || (v12 = v8 & 0x7FFFFFFF,
         v13 = v10 & 0x7FFFFFFF,
         v14 = HIDWORD(v6) & 0x7FFFFFFF,
         v15 = HIDWORD(v9) & 0x7FFFFFFF,
         (_DWORD)v6 == (_DWORD)v9)
     && (v14 > v15 || v14 == v15 && (v12 > v13 || (HIDWORD(v8) & 0x7FFFFFFFu) > (v11 & 0x7FFFFFFFu) && v12 == v13))) )
  {
    *a1 = (__int64)(a1 + 2);
    sub_CC3EF0(a1, *(_BYTE **)a2, *(_QWORD *)a2 + *(_QWORD *)(a2 + 8));
  }
  else
  {
    *a1 = (__int64)(a1 + 2);
    sub_CC3EF0(a1, *(_BYTE **)a3, *(_QWORD *)a3 + *(_QWORD *)(a3 + 8));
  }
  return a1;
}
