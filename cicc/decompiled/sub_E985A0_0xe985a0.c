// Function: sub_E985A0
// Address: 0xe985a0
//
unsigned __int64 __fastcall sub_E985A0(_DWORD *a1, __int64 a2, __int64 a3)
{
  int v3; // edx
  int v4; // rdx^4
  unsigned int v6; // esi
  unsigned int v7; // r9d
  unsigned int v8; // r8d
  unsigned int v9; // edi
  unsigned __int64 v12; // [rsp+10h] [rbp-10h]

  v12 = sub_CC8340(a1);
  if ( (unsigned int)v12 | v4 & 0x7FFFFFFF | (HIDWORD(v12) | v3) & 0x7FFFFFFF
    && ((unsigned int)v12 > (unsigned int)a2
     || (v6 = HIDWORD(v12) & 0x7FFFFFFF,
         v7 = v3 & 0x7FFFFFFF,
         v8 = a3 & 0x7FFFFFFF,
         v9 = HIDWORD(a2) & 0x7FFFFFFF,
         (_DWORD)v12 == (_DWORD)a2)
     && (v6 > v9 || v6 == v9 && (v7 > v8 || (v4 & 0x7FFFFFFFu) > (HIDWORD(a3) & 0x7FFFFFFFu) && v7 == v8))) )
  {
    return v12;
  }
  else
  {
    return a2;
  }
}
