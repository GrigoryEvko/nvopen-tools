// Function: sub_7C9DD0
// Address: 0x7c9dd0
//
__int64 sub_7C9DD0()
{
  __int64 v0; // rbx
  const char **v1; // r14
  const char *v2; // r12
  __int64 v3; // rcx
  __m128i *v4; // rsi
  __int64 result; // rax
  __int64 v6; // [rsp-10h] [rbp-80h]
  size_t v7; // [rsp+10h] [rbp-60h]
  __int64 v8; // [rsp+18h] [rbp-58h]
  char v9; // [rsp+2Ch] [rbp-44h] BYREF
  __int64 v10; // [rsp+30h] [rbp-40h] BYREF
  char v11; // [rsp+38h] [rbp-38h] BYREF

  unk_4F061E8 = sub_823970(832);
  v0 = 416;
  v1 = (const char **)&off_4B6EAF0;
  do
  {
    v2 = *v1;
    v7 = strlen(*v1);
    sub_7B0E60(v7 + 6);
    v3 = qword_4F06498;
    *(_BYTE *)qword_4F06498 = 34;
    v8 = v3;
    strcpy((char *)(v3 + 1), v2);
    *(_BYTE *)(v8 + v7 + 1) = 34;
    *(_BYTE *)(v8 + v7 + 2) = 0;
    *(_BYTE *)(v8 + v7 + 3) = 2;
    *(_BYTE *)(v8 + v7 + 4) = 0;
    *(_BYTE *)(v8 + v7 + 5) = 1;
    *(_DWORD *)&word_4F06480 = 0;
    qword_4F06410 = (const char *)qword_4F06498;
    qword_4F06460 = (_BYTE *)(qword_4F06498 + 1LL);
    v10 = 0;
    if ( (unsigned int)sub_7B6B00(&v10, 0, 17, 34, 0, -1, (_BYTE *)qword_4F06498, 0) )
      sub_721090();
    ++v1;
    sub_7CE2C0((_DWORD)qword_4F06410 + 1, qword_4F06408, 17, v10, (unsigned int)&v9, (unsigned int)&v11, 0);
    v4 = (__m128i *)(v0 + unk_4F061E8);
    v0 += 208;
    sub_72A510(xmmword_4F06300, v4);
    qword_4F06460 = (_BYTE *)(qword_4F06498 + v7 + 4);
    result = v6;
  }
  while ( v0 != 832 );
  return result;
}
