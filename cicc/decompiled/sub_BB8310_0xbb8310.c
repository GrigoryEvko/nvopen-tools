// Function: sub_BB8310
// Address: 0xbb8310
//
__int64 __fastcall sub_BB8310(__int64 a1, int *a2)
{
  int v2; // r12d
  int *v3; // rbx
  __int64 result; // rax
  int *v5; // r12
  _BOOL4 v6; // r15d
  __int64 v7; // rax
  __int64 v8; // rax
  int *v9; // rdx
  int *v10; // r13
  int v11[13]; // [rsp+Ch] [rbp-34h] BYREF

  v2 = *a2;
  if ( !byte_4F82228 && (unsigned int)sub_2207590(&byte_4F82228) )
  {
    qword_4F82288 = (__int64)&dword_4F82278;
    qword_4F82290 = (__int64)&dword_4F82278;
    qword_4F82240 = (__int64)&unk_49DACB8;
    qword_4F82250 = (__int64)&unk_4F82260;
    qword_4F82248 = 0x7FFFFFFF;
    qword_4F82258 = 0x400000000LL;
    dword_4F82278 = 0;
    qword_4F82280 = 0;
    qword_4F82298 = 0;
    __cxa_atexit((void (*)(void *))sub_BB7700, &dword_4F82278 - 14, &qword_4A427C0);
    sub_2207640(&byte_4F82228);
  }
  v11[0] = v2;
  if ( qword_4F82298 )
    return sub_BB8160((__int64)&unk_4F82270, v11);
  v3 = (int *)(qword_4F82250 + 4LL * (unsigned int)qword_4F82258);
  if ( (int *)qword_4F82250 != v3 )
  {
    result = qword_4F82250;
    while ( v2 != *(_DWORD *)result )
    {
      result += 4;
      if ( v3 == (int *)result )
        goto LABEL_14;
    }
    if ( v3 != (int *)result )
      return result;
LABEL_14:
    if ( (unsigned int)qword_4F82258 <= 3uLL )
      goto LABEL_15;
    v5 = (int *)qword_4F82250;
    do
    {
      v8 = sub_BB8210((_QWORD *)&dword_4F82278 - 1, (__int64)&dword_4F82278, v5);
      v10 = v9;
      if ( v9 )
      {
        v6 = v8 || v9 == &dword_4F82278 || *v5 < v9[8];
        v7 = sub_22077B0(40);
        *(_DWORD *)(v7 + 32) = *v5;
        sub_220F040(v6, v7, v10, &dword_4F82278);
        ++qword_4F82298;
      }
      ++v5;
    }
    while ( v3 != v5 );
    goto LABEL_10;
  }
  if ( (unsigned int)qword_4F82258 > 3uLL )
  {
LABEL_10:
    LODWORD(qword_4F82258) = 0;
    return sub_BB8160((__int64)&unk_4F82270, v11);
  }
LABEL_15:
  result = HIDWORD(qword_4F82258);
  if ( (unsigned __int64)(unsigned int)qword_4F82258 + 1 > HIDWORD(qword_4F82258) )
  {
    sub_C8D5F0((char *)&unk_4F82260 - 16, &unk_4F82260, (unsigned int)qword_4F82258 + 1LL, 4);
    result = qword_4F82250;
    v3 = (int *)(qword_4F82250 + 4LL * (unsigned int)qword_4F82258);
  }
  *v3 = v2;
  LODWORD(qword_4F82258) = qword_4F82258 + 1;
  return result;
}
